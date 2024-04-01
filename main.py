import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import argparse
import sys
from models.misc import is_dist_avail_and_initialized, save_on_master, init_distributed_mode
from models.Dataloader import ClipCocoDataset, CaptionDataset, id_collate
from models.models import build_model
from models.scheduler import CosineAnnealingLRWarmup

from models.engine import train_one_epoch, eval_caption
import json
import os

os.environ['CURL_CA_BUNDLE'] = ''


def train_decoder(dataset: ClipCocoDataset, args,
                  lr: float = 1e-4, warmup_steps: int = 200, output_dir: str = ".", output_prefix: str = "",
                  test_dataset=None, val_dataset=None, min_lr=0.1):
    # device = torch.device('cuda:1')
    batch_size = args.bs
    epochs = args.epochs

    if args.use_ddp == 1:
        init_distributed_mode(args)
    else:
        args.distributed = False
        args.rank = 0
        args.gpu = 0

    print(args)
    print('****************')
    # print(model)
    print(args.model_name)
    print('****************')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    args.is_master = args.rank == 0

    # set the device
    # torch.cuda.set_device(args.rank)
    device = torch.device('cuda:' + str(args.gpu))
    SEED = 42
    torch.cuda.manual_seed_all(SEED)

    model = build_model(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    model.to(device)
    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = DDP(
            model,
            device_ids=[args.gpu],
            # output_device=args.rank,
            find_unused_parameters=False
        )
        sampler = DistributedSampler(dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
        sampler_test = DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                                  drop_last=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, sampler=sampler_test, batch_size=batch_size,
                                 drop_last=False, num_workers=args.num_workers, collate_fn=id_collate)

    val_dataloader = DataLoader(val_dataset, sampler=sampler_val, batch_size=batch_size,
                                drop_last=False, num_workers=args.num_workers, collate_fn=id_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
                                           warmup_iter=warmup_steps,
                                           warmup_ratio=0.01,
                                           T_max=args.epochs - 1,
                                           eta_min=min_lr)

    if args.resume and os.path.exists(args.resume):
        print(f'Resume from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' not in checkpoint:
            model_without_ddp.load_state_dict(checkpoint)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f'Pretrained from {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model' not in checkpoint:
            model_without_ddp.load_state_dict(checkpoint)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        if is_dist_avail_and_initialized():
            eval_caption(model.module, output_dir, args, test_dataloader, device, f'test', split='test')
            torch.distributed.barrier()
        else:
            eval_caption(model, output_dir, args, test_dataloader, device, f'test', split='test')

        return

    for epoch in range(args.start_epoch, epochs):
        sys.stdout.flush()
        progress = None
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader) / 10), desc=output_prefix)

        if is_dist_avail_and_initialized():
            dist.barrier()
            sampler.set_epoch(epoch)

        train_one_epoch(train_dataloader, device, model, optimizer, lr_scheduler, args, progress, val_dataloader,
                        output_dir, epoch)

        lr_scheduler.step()
        if args.is_master:
            log_dir = os.path.join(output_dir, args.dataset + '.txt')
            with open(log_dir, 'a+') as f:
                f.writelines('epoch ' + str(epoch) + ': ' + progress.postfix + '\r\n')
            progress.close()

            model_cpt = model_without_ddp.state_dict()
            for i in list(model_cpt.keys()):
                if 'clip_model.' in i and 'model.model.' in i:
                    del model_cpt[i]
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
                save_on_master({
                    'model': model_cpt,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.is_master and len(progress) != int(len(train_dataloader) / 10):
            progress = tqdm(total=int(len(train_dataloader) / 10), desc=output_prefix)
            progress.set_postfix({"loss_token": -1, "acc_token": -1})
            progress.update()

        if (epoch + 1) % args.eval_interval == 0:
            if is_dist_avail_and_initialized():
                eval_caption(model.module, output_dir, args, val_dataloader, device, f'epoch_end_{epoch}', split='val')
                torch.distributed.barrier()
            else:
                eval_caption(model, output_dir, args, val_dataloader, device, f'epoch_end_{epoch}', split='val')

    if is_dist_avail_and_initialized():
        eval_caption(model.module, output_dir, args, test_dataloader, device, f'test', split='test')
        torch.distributed.barrier()
    else:
        eval_caption(model, output_dir, args, test_dataloader, device, f'test', split='test')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/doubleMTA_lr1e5_ignore0.pkl')
    parser.add_argument('--out_dir', default='./coco_model')
    parser.add_argument('--prefix', default='./coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='coco', help='coco or cc3m or bookcorpus')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--prefix_length_clip', type=int, default=1)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    # new settings
    parser.add_argument('--language_model', default='facebook/opt-125m')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--use_ddp', default=1)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--data_ids', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--pretrained', default='')

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--model_name', default='MacCap')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=0.1, type=float)
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--num_query', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--eval_type', default='caption', choices=['caption', 'retrieval', 'token_analysis', 'vqa'])
    # eval caption settings
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--decoding_len', default=16, type=int)
    parser.add_argument('--k', default=45, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--test_dataset', default='coco')
    parser.add_argument('--test_path', default='./data/mscoco/mscoco_test.json')
    parser.add_argument('--test_image_prefix_path', default='./data/mscoco/test_images/')
    parser.add_argument('--val_path', default='./data/mscoco/mscoco_val.json')
    parser.add_argument('--val_image_prefix_path', default='./data/coco/images/val2014/')
    parser.add_argument('--generation_strategy', default='naive',
                        choices=['naive', 'text', 'magic', 'vqa', 'vqa_demo', 'multi_noise', 'contrastive',
                                 'visualization'])
    parser.add_argument('--eval_desc', default='')
    parser.add_argument('--num_decoder_layer', default=1, type=int)

    # memory bank parameters
    parser.add_argument('--memory_path', default='./data/decap_cc3m.pth')
    # memory bank during inference
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--inference_mb_type', default='', choices=['sequence', 'pooling', 'cls_only', ''])

    # others
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--eval_inside_epoch', action='store_true')
    parser.add_argument('--save_each', action='store_true')

    # noise injected
    parser.add_argument('--noise_variance', default=0.016, type=float)
    parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'vonMisesFisher'])

    # multi-noise during inference generation
    parser.add_argument('--num_noise', default=1, type=int)
    parser.add_argument('--noise_k', default=0.016, type=float, help='noise variance during inference')
    parser.add_argument('--noise_N_var', default=0., type=float, nargs='+',
                        help='noise variance for generate multiple text in reconstruction sampling')
    parser.add_argument('--num_reconstruction', default=10, type=int)
    parser.add_argument('--sampling_type', default='naive',
                        choices=['features', 'mix_features', 'weighted_mix_features', 'sentences', 'repeat',
                                 'text_repeat', 'reconstruction', 'reconstruction_rank', 'reconstruction_repeat',
                                 'reconstruction_concat', 'reconstruction_concat_wo_token_noise', 'naive'])

    # LAVIS
    parser.add_argument("--cfg-path", required=False, help="path to configuration file.", default='')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument('--vqa_text_only', action='store_true')
    parser.add_argument('--vqa_caption_text', action='store_true')
    parser.add_argument('--demonstration', action='store_true')
    parser.add_argument('--demo_path', default='./tools/demonstration/train2014.json', type=str)
    parser.add_argument('--demo_image_prefix_path', default='./data/coco/images/train2014/')

    parser.add_argument('--demo_num', default=1, type=int)
    parser.add_argument('--infer_patch_weight', default=1.0, type=float)
    parser.add_argument('--infer_window_size', default=5, type=int)
    parser.add_argument('--infer_variance_ratio', default=0.1, type=float)
    parser.add_argument('--caption_save_path', default='', type=str)
    parser.add_argument('--train_seq_length', default=10, type=int)
    parser.add_argument('--mask_ratio', default=0.15, type=float)

    parser.add_argument('--train_max_length', default=77, type=int)

    # contrastive loss
    parser.add_argument('--contrastive_loss', action='store_true')
    parser.add_argument('--contrastive_loss_weight', default=1.0, type=float)
    parser.add_argument('--infer_patch_selection', action='store_true')
    parser.add_argument('--infer_patch_attn_weight', action='store_true')
    parser.add_argument('--infer_multi_cls', action='store_true')

    # contrastive generation
    parser.add_argument('--contrastive_generation_diversity', default=4.0, type=float)
    parser.add_argument('--contrastive_generation_beam_size', default=4, type=int)
    parser.add_argument('--generation_multi', action='store_true')
    parser.add_argument('--generation_text', action='store_true')
    parser.add_argument('--vis_path', default='')
    parser.add_argument('--ft_llm', action='store_true')

    # VQA training
    parser.add_argument('--generate_vqa', action='store_true')
    parser.add_argument('--train_vqa', action='store_true')
    parser.add_argument('--vqa_image', default='./data/mscoco/test_images/', type=str)
    parser.add_argument('--transformer_clip_length', default=4, type=int)

    parser.add_argument('--train_caption', action='store_true')

    # train with unlabel image
    parser.add_argument('--train_w_image', action='store_true')
    parser.add_argument('--image_with_gt', action='store_true')
    parser.add_argument('--image_mode', type=str, choices=['mix', 'image_only', 'file'])
    parser.add_argument('--train_path', default='./data/mscoco/mscoco_train.json')
    parser.add_argument('--train_image_prefix_path', default='./data/coco/images/train2014/')
    parser.add_argument('--multi_cap', default=5, type=int)
    parser.add_argument('--img_frac', default=0.1, type=float)
    parser.add_argument('--img_noise_variance', default=0., type=float)

    parser.add_argument('--eval_interval', default=1, type=int)


    args = parser.parse_args()

    dataset = ClipCocoDataset('data/' + args.dataset + '_train.json', args.language_model, args.data_ids,
                                  args.train_max_length)

    test_dataset = CaptionDataset(args, 'test')
    val_dataset = CaptionDataset(args, 'val')
    print('Datasets generated!')
    train_decoder(dataset, args, output_dir=args.out_dir, output_prefix=f'{args.dataset}_{args.out_dir}', lr=args.lr,
                  test_dataset=test_dataset, val_dataset=val_dataset, min_lr=args.min_lr, warmup_steps=args.warmup_steps)


if __name__ == '__main__':
    main()
