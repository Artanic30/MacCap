import random

import torch
import os
import json
from tqdm import tqdm
from models.misc import is_dist_avail_and_initialized, is_main_process, get_rank
import numpy as np
import re
import random
from ClipContextual import clip
from PIL import Image
import torch.nn.functional as F
from torch.cuda.amp import GradScaler


def train_one_epoch(train_dataloader, device, model, optimizer, lr_scheduler, args, progress=None, val_dataloader=None,
                    output_dir='', epoch='2'):
    scaler = GradScaler()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    loss_token_save, ac_save = {}, 0
    for idx, item in enumerate(train_dataloader):
        if len(item) == 2:
            clip_tokens, llm_tokens = item
            clip_tokens, llm_tokens = clip_tokens.to(device), llm_tokens.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output, ac = model(clip_tokens, llm_tokens)
        elif len(item) == 3:
            clip_tokens, llm_tokens, is_image = item
            clip_tokens, llm_tokens = clip_tokens.to(device), llm_tokens.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                forward_output = model(clip_tokens, llm_tokens, is_image)
                if isinstance(forward_output, dict):
                    output = forward_output
                    ac = torch.Tensor([0])
                else:
                    output, ac = forward_output
        else:
            raise ValueError

        loss_token = sum(list(output.values()), start=0)
        loss_token = loss_token / gradient_accumulation_steps

        scaler.scale(loss_token).backward()
        if (idx + 1) % gradient_accumulation_steps == 0:
            # print(f'idx: {idx}, gda step')

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        lr_scheduler.iter_step()

        if args.is_master and progress is not None:
            if (idx + 1) % 10 == 0:
                logger = {"acc_token": ac_save / 10.0, 'lr': [round(group['lr'], 8) for group in optimizer.param_groups]}
                for k, v in loss_token_save.items():
                    logger.update({
                        k: v / 10.0
                    })

                progress.set_postfix(logger)
                progress.update()
                loss_token_save, ac_save = {}, 0
            else:
                for k, v in output.items():
                    if k not in loss_token_save:
                        loss_token_save[k] = v.item()
                    else:
                        loss_token_save[k] += v.item()
                ac_save += ac.item()

        if args.eval_inside_epoch and (idx + 1) % int(len(train_dataloader) / 2 + 1) == 0:
            if is_dist_avail_and_initialized():
                eval_caption(model.module, output_dir, args, val_dataloader, device, f'{epoch}_{idx}', split='val')
                torch.distributed.barrier()
            else:
                eval_caption(model, output_dir, args, val_dataloader, device, f'{epoch}_{idx}', split='val')


@torch.no_grad()
def eval_caption(model, output_dir, args, val_dataloader, device, epoch='last', split='test'):
    model.eval()
    save_path = os.path.join(output_dir,
                             args.test_dataset + f'_{epoch}_{args.generation_strategy}_{args.decoding_len}'
                                                 f'{"_" + args.inference_mb_type if args.inference_mb_type else ""}'
                                                 f'_{args.temperature}.json'
                             if not args.caption_save_path else args.caption_save_path)
    print(f'Saving to {save_path}')
    tmp_path = os.path.join(output_dir, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)
    print(split)
    if split == 'test':
        with open(args.test_path, 'r') as f:
            results = json.load(f)
    elif split == 'val':
        with open(args.val_path, 'r') as f:
            results = json.load(f)

    results = {i['image_name']: i for i in results}
    count = 0

    for image, file_path in tqdm(val_dataloader, desc='Evaluation results: '):
        # for image, file_path in val_dataloader:
        image = image.to(device)
        select_idx = []
        for i in range(len(file_path)):
            if 'prediction' not in results[file_path[i]]:
                select_idx.append(i)

        if not select_idx:
            continue
        else:
            image = image[select_idx]
            file_path = [file_path[i] for i in select_idx]

        if args.generation_multi:
            out_list = [[] for _ in range(image.shape[0])]
            for i in range(args.contrastive_generation_beam_size):
                output_text = model.generate(image, args.generation_strategy)
                for idx, t in enumerate(output_text):
                    out_list[idx].append(t)
            output_text = out_list
        elif args.generation_text:
            # from ClipContextual import clip
            out_list = [[] for _ in range(image.shape[0])]
            output_text = model.generate(image, args.generation_strategy)
            for idx, t in enumerate(output_text):
                out_list[idx].append(t)

            for i in range(args.contrastive_generation_beam_size - 1):
                c_t_l = []
                for t in output_text:
                    c_t_l.append(clip.tokenize(t[:80]))
                clip_tokens = torch.cat(c_t_l).to(image.device)
                output_text = model.generate(image, args.generation_strategy, clip_tokens=clip_tokens)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output_text = model.generate(image, args.generation_strategy)

        if args.generation_multi:
            clip_model = model.clip_model

            image_features, _, _ = clip_model.visual(image.half())
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for idx in range(len(output_text)):
                c_t_l = []
                for t in output_text[idx]:
                    c_t_l.append(clip.tokenize(t[:80]))
                clip_tokens = torch.cat(c_t_l)
                text_features, _, _ = clip_model.encode_text(clip_tokens.cuda())
                text_features /= text_features.norm(dim=-1, keepdim=True)
                score = (image_features[idx] @ text_features.T).softmax(dim=-1)
                select_idx = score.topk(1).indices
                select_text = output_text[idx][select_idx]
                results[file_path[idx]]['prediction'] = select_text
                results[file_path[idx]]['clip_score'] = [(t, round(i.item(), 3)) for i, t in
                                                         zip(score, output_text[idx])]
        else:
            if isinstance(output_text, list):
                for idx, i in enumerate(output_text):
                    results[file_path[idx]]['prediction'] = i
            else:
                results[file_path[0]]['prediction'] = output_text

        count += 1
        if count % 10 == 0 or count == len(val_dataloader):
            if not is_dist_avail_and_initialized():
                results_save = [v for k, v in results.items()]
                with open(save_path, 'w') as f:
                    f.write(json.dumps(results_save))
            else:
                with open(os.path.join(tmp_path, args.test_dataset + f'rat_{get_rank()}.json'), 'w') as f:
                    f.write(json.dumps(results))
                torch.distributed.barrier()

                if is_main_process():
                    json_files = os.listdir(tmp_path)
                    valid_files = []
                    for i in json_files:
                        if 'rat_' in i and '.json' in i:
                            valid_files.append(i)
                    collected_rationale = {}
                    for i in valid_files:
                        with open(f'{tmp_path}/{i}', 'r') as f:
                            rat_rank = json.loads(f.read())
                        if not collected_rationale:
                            collected_rationale = rat_rank
                        else:
                            for k, v in rat_rank.items():
                                if 'prediction' not in collected_rationale[k]:
                                    collected_rationale[k] = v

                    results_save = [v for k, v in collected_rationale.items()]
                    with open(save_path, 'w') as f:
                        f.write(json.dumps(results_save))
