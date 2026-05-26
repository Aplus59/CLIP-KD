#!/bin/bash
cd src

TEACHER_CKPT="../logs/best_teacher_wd02.pt"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Error: Teacher checkpoint not found at $TEACHER_CKPT"
    echo "Please copy your best epoch file to logs/best_teacher.pt"
    exit 1
fi

export USE_AUG=1

METHODS=("AFD" "CLIP-KD")
LR=1e-5
WD=0.2

for METHOD in "${METHODS[@]}"; do
    echo "========================================="
    echo "Running Distillation Method: $METHOD"
    echo "========================================="
    
    CKD=0; ICL=0; CROSS_KD=0; FD=0; GD=0; AFD=0
    
    case $METHOD in
        "AFD")     AFD=1.0 ;;
        "CLIP-KD") CKD=1.0; ICL=1.0; FD=2000.0 ;;
    esac

    TAG="fathomnet-distill-${METHOD}-student-vit-t"
    
    /workspace/CLIP-KD/venv/bin/torchrun --nproc_per_node 1 -m \
        training.main_kd \
        --save-frequency 0 \
        --zeroshot-frequency 1 \
        --report-to tensorboard \
        --train-data="../newd/db/train.csv"  \
        --val-data="../newd/db/test.csv"  \
        --data-root ../newd/images/ \
        --val-data-root ../newd/images/ \
        --csv-img-key file_name \
        --csv-caption-key text \
        --csv-separator "," \
        --warmup 200 \
        --batch-size=64 \
        --lr=${LR} \
        --wd=${WD} \
        --epochs 20 \
        --workers=4 \
        --model ViT-T-16 \
        --pretrained /workspace/CLIP-KD/checkpoints/ViT-B-16_teacher/baselines/ViT_T_16_cc3m_12m_ep32.pt \
        --t-model ViT-B-16 \
        --t-model-checkpoint ${TEACHER_CKPT} \
        --logs ../logs/ \
        --alpha_ckd_loss ${CKD} \
        --alpha_icl_loss ${ICL} \
        --alpha_cross_kd_loss ${CROSS_KD} \
        --alpha_fd_loss ${FD} \
        --alpha_gd_loss ${GD} \
        --alpha_afd_loss ${AFD} \
        --tag ${TAG}

    # Clean up checkpoints after each method to save disk space
    echo "Cleaning up checkpoints for $METHOD..."
    rm -f ../logs/*-tag_${TAG}/checkpoints/epoch_*.pt
done
