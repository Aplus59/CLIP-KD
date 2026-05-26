#!/bin/bash
cd src

# Hyperparameter grid
LEARNING_RATES=(1e-4 5e-4 1e-5)
WEIGHT_DECAYS=(0.1 0.2)

for LR in "${LEARNING_RATES[@]}"; do
    for WD in "${WEIGHT_DECAYS[@]}"; do
        TAG="fathomnet-hp-lr${LR}-wd${WD}"
        echo "Running: LR=${LR}, WD=${WD}"
        
        torchrun --nproc_per_node 1 -m \
            training.main \
            --save-frequency 1 \
            --zeroshot-frequency 1 \
            --report-to tensorboard \
            --train-data="../newd/db/train_split.csv"  \
            --val-data="../newd/db/val_split.csv"  \
            --data-root ../newd/images/ \
            --val-data-root ../newd/images/ \
            --csv-img-key file_name \
            --csv-caption-key text \
            --csv-separator "," \
            --warmup 1000 \
            --batch-size=128 \
            --lr=${LR} \
            --wd=${WD} \
            --epochs 32 \
            --workers=4 \
            --model ViT-B-16 \
            --logs ../logs/ \
            --tag ${TAG}
    done
done
