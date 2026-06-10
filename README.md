# CLIP-KD: Marine Domain Distillation

This repository contains the code for adapting CLIP model distillation to the marine domain. This is a Master 1 project (Vision et Machines Intelligentes) building upon the original [CLIP-KD](https://github.com/winycg/CLIP-KD) architecture.

**Authors:** Ho Bao Khanh NGUYEN, Xiangrui FENG  
**Supervisors:** M. Ayoub, M. Camille  

## Dataset Preparation

We built a marine equivalent to CC3M/12M using the *Mini FathomNet Out of Sample Detection* dataset. Since marine datasets lack dense captions, we used an automated pipeline:

1. **Contextual Cropping:** We used YOLOv5 bounding boxes to crop the main species with a 30% margin to include the habitat.
2. **VLM Captioning:** We used GPT-4o-mini to generate rich captions based on validated species metadata.

**Download Dataset:** [fantom.zip - Google Drive](https://drive.google.com/file/d/1xOi3OyDi4jBP872YBxPPai4_p2GoKKNy/view?usp=sharing) 

* **Train set:** 1,186 images (`train.csv`)
* **Test set:** 45 images (`test.csv`)

## Hardware & Training Environment

* **GPU:** 1x NVIDIA RTX 5060 Ti (16GB VRAM)
* **CPU:** AMD EPYC 7402 (24 cores)
* **Framework:** PyTorch (CUDA 13.2)

## Teacher Fine-Tuning

The Teacher model is initialized with CC3M+12M weights and fine-tuned on our marine dataset using Transfer Learning. We froze the weights at **Epoch 23** as it provided the best modal equilibrium.

| Role | Network | Text-to-Image R@1 | Image-to-Text R@1 | Mean R@1 | 
| :----: | :----: | :----: | :----: | :----: |
| Teacher | ViT-B/16 | 42.22% | 42.22% | 42.22% |

* **Hyperparameters:** Batch Size 128, LR 1e-3, Weight Decay 0.1, 32 Epochs.

## Distill CLIP models (Student)

The Student model is supervised by the ViT-B/16 Teacher (Epoch 23) and distilled on our marine dataset. 

| Role | Network | Method | Text-to-Image R@1 | Mean R@1 | Mean R@10 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Student | ViT-T/16 | CKD | 28.89% | 24.44% | > 73.00% |
| Student | ViT-T/16 | CLIP-KD (Combined) | - | 15.56% | - |

* **Hyperparameters:** Batch Size 64, LR 1e-5, Weight Decay 0.2, 20 Epochs.

> **Key Observation:** The Contrastive Knowledge Distillation (CKD) strategy alone outperformed the combined CLIP-KD method. The large hyperparameters of the combined method caused severe overfitting on our small dataset. CKD successfully compressed the model while preserving >73% R@10 accuracy, making it viable for AUV deployment.
