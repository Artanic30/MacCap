python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port 29403 \
        --use_env \
        main.py \
        --out_dir ./output/coco \
        --dataset coco \
        --epochs 8 \
        --language_model facebook/opt-1.3b \
        --model_name MacCap \
        --lr 1e-4 \
        --bs 16