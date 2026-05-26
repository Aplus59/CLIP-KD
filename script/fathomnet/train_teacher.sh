#!/bin/bash
cd src
torchrun --nproc_per_node 1 -m \
    training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="../newd/db/train.csv"  \
    --val-data="../newd/db/test.csv"  \
    --data-root ../newd/images/ \
    --val-data-root ../newd/images/ \
    --csv-img-key file_name \
    --csv-caption-key text \
    --csv-separator "," \
    --warmup 1000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=4 \
    --model ViT-B-16 \
    --pretrained ../checkpoints/ViT-B-16_teacher/ViT_B_16_cc3m_12m_ep32.pt \
    --logs ../logs/ \
    --tag fathomnet-teacher-vit-b
