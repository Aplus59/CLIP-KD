#!/bin/bash
cd src
# Adjust --t-model-checkpoint if the teacher model file is named differently
torchrun --nproc_per_node 1 -m \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="../newd/db/train.csv"  \
    --val-data="../newd/db/test.csv"  \
    --data-root ../newd/images/ \
    --val-data-root ../newd/images/ \
    --csv-img-key file_name \
    --csv-caption-key text \
    --warmup 1000 \
    --batch-size=64 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=4 \
    --model ViT-T-16 \
    --t-model ViT-B-16 \
    --t-model-checkpoint ../logs/fathomnet-teacher-vit-b/checkpoints/epoch_32.pt \
    --logs ../logs/ \
    --alpha_ckd_loss 1. \
    --tag fathomnet-distill-crd-vit-t
