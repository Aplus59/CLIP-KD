#!/bin/bash
cd src

LEARNING_RATES=(1e-5 5e-6 1e-6)
WEIGHT_DECAYS=(0.1 0.2)

export USE_AUG=1

for LR in "${LEARNING_RATES[@]}"; do
    for WD in "${WEIGHT_DECAYS[@]}"; do
        TAG="fathomnet-transfer-aug-lr${LR}-wd${WD}"
        echo "Running Transfer Learning (With Aug): LR=${LR}, WD=${WD}"
        
        torchrun --nproc_per_node 1 -m \
            training.main \
            --pretrained /workspace/CLIP-KD/checkpoints/ViT-B-16_teacher/ViT_B_16_cc3m_12m_ep32.pt \
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
            --warmup 200 \
            --batch-size=128 \
            --lr=${LR} \
            --wd=${WD} \
            --epochs 20 \
            --workers=4 \
            --model ViT-B-16 \
            --logs ../logs/ \
            --tag ${TAG}
    done
done
