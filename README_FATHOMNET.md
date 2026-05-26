# FathomNet CLIP-KD: User Guide

This project applies the CLIP-KD distillation method on an underwater dataset (FathomNet). 

## 1. Prerequisites and Installation

**Installing dependencies:**
Ensure your environment is properly configured.
```bash
pip install -r requirements-training.txt
pip install -r requirements-test.txt
pip install -r requirements.txt
```
**Downloading Dataset FathomNet from gg drive:**
Download, unzip and put it into CLIP-KD/newd.
```bash
pip install gdown
gdown 1xOi3OyDi4jBP872YBxPPai4_p2GoKKNy
```
**Downloading Pre-trained Weights (Checkpoints):**
Both Teacher and Student models require the CC3M+12M baseline weights to start. Execute the following commands to create the necessary directories and download the original `.pt` files:
```bash
mkdir -p checkpoints/ViT-B-16_teacher/baselines

# Download the baseline model for the Teacher (ViT-B/16)
wget -O checkpoints/ViT-B-16_teacher/ViT_B_16_cc3m_12m_ep32.pt https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.pt

# Download the baseline model for the Student (ViT-T/16)
wget -O checkpoints/ViT-B-16_teacher/baselines/ViT_T_16_cc3m_12m_ep32.pt https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt
```

## 2. Data Structure
The processed data must be located in:
- Images: `newd/images/`
- CSV files: `newd/db/`
  - `train.csv` (1186 pairs)
  - `test.csv` (45 pairs)
  - `train_split.csv` / `val_split.csv` (generated via `split_train_val.py` for distillation)

---

## 3. Training the Teacher Model (ViT-B/16)

To fine-tune the Teacher model on the marine data, run the following script:
```bash
bash script/fathomnet/train_teacher.sh
```
- The script uses `ViT-B-16` initialized with CC3M+12M pre-trained weights.
- It trains for 32 epochs (batch size 128) using `train.csv` as training data and `test.csv` as validation data.
- The results and checkpoints will be saved in a sub-directory within `logs/`.

---

## 4. Extracting the Best Teacher Model and Visualization

Once training is complete, you need to extract the best checkpoint (Epoch 23 was identified as optimal) to guide the distillation process.

**Copy the best checkpoint for distillation:**
```bash
# Replace the directory name with the one generated in your logs/ folder
cp logs/2026_05_26-18_23_35-model_ViT-B-16-lr_0.001-b_128-epochs_32-tag_fathomnet-teacher-vit-b/checkpoints/epoch_23.pt logs/best_teacher_wd02.pt
```
*(Note: If you want to clean up your hard drive after this step, you can delete the other heavy checkpoints using `rm -f logs/*/checkpoints/epoch_*.pt`. This will not delete the `best_teacher_wd02.pt` file you just copied).*

**Generate the Performance Graph (Plot):**
To generate an image illustrating the evolution of the Validation Loss and Recall@1:
```bash
python plot_results.py
```
The script will automatically read the `results.jsonl` log file and create an image named `teacher_training_plot.png`. You can then download or view this image to include it in your report.

---

## 5. Distillation to the Student Model (ViT-T/16)

Once the `logs/best_teacher_wd02.pt` file is ready, you can launch the distillation using 6 different strategies (CrossKD, ICL, GD, FD, CKD, AFD):
```bash
bash script/fathomnet/run_all_distill.sh
```
- The script launches distillation to a `ViT-T-16` model.
- Training is performed for 20 epochs for each method.
- **Attention:** The script automatically deletes the student model weights (`.pt`) at the end of each method to save disk space. Only the evaluation log files (`results.jsonl`) are kept for analysis.

---

## 6. Comparative Evaluation of Distillation (Table)

To automatically extract the best Recall@1, R@5, R@10 scores of each distillation method and display a clear summary table:
```bash
python compare_distill.py
```
This script will output a comprehensive table in the terminal, which you can directly copy-paste into your scientific report to compare the effectiveness of the methods (e.g., AFD vs GD).
