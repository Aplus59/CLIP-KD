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

| Method | Epoch | Val Loss | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Avg R@1 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| CKD | 14 | 3.5946 | 20.00% | 57.78% | 66.67% | 28.89% | 60.00% | 73.33% | 24.44% |
| GD | 12 | 3.5024 | 20.00% | 55.56% | 68.89% | 26.67% | 55.56% | 75.56% | 23.33% |
| FD | 12 | 3.5002 | 20.00% | 55.56% | 68.89% | 26.67% | 55.56% | 75.56% | 23.33% |
| AFD | 11 | 3.4813 | 20.00% | 53.33% | 66.67% | 26.67% | 53.33% | 73.33% | 23.33% |
| CrossKD | 12 | 3.5074 | 17.78% | 55.56% | 66.67% | 26.67% | 55.56% | 77.78% | 22.22% |
| ICL | 12 | 3.5105 | 17.78% | 55.56% | 66.67% | 26.67% | 55.56% | 77.78% | 22.22% |
| CLIP-KD | 9 | 3.5437 | 17.78% | 48.89% | 71.11% | 13.33% | 46.67% | 68.89% | 15.56% |

* **Hyperparameters:** Batch Size 64, LR 1e-5, Weight Decay 0.2, 20 Epochs.

> **Key Observation:** The Contrastive Knowledge Distillation (CKD) strategy alone outperformed the combined CLIP-KD method. The large hyperparameters of the combined method caused severe overfitting on our small dataset. CKD successfully compressed the model while preserving >73% R@10 accuracy, making it viable for AUV deployment.
