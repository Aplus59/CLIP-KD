import json
import matplotlib.pyplot as plt
import os
import glob

# Automatically find the latest log directory containing teacher's results.jsonl
def find_latest_log_dir():
    log_dirs = glob.glob("logs/*teacher*")
    log_dirs.sort(key=os.path.getmtime, reverse=True)
    for d in log_dirs:
        if os.path.exists(os.path.join(d, "checkpoints", "results.jsonl")):
            return os.path.join(d, "checkpoints", "results.jsonl")
    return None

def main():
    log_file = find_latest_log_dir()

    if log_file is None:
        print("results.jsonl file not found in logs/ directory")
        return

    print(f"Reading data from: {log_file}")

    epochs = []
    val_loss = []
    i2t_r1 = []
    t2i_r1 = []

    # Read data from log
    with open(log_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            epochs.append(data["epoch"])
            val_loss.append(data["val_loss"])
            i2t_r1.append(data["image_to_text_R@1"])
            t2i_r1.append(data["text_to_image_R@1"])

    # Plot the graph
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis: Validation Loss (Red color)
    color1 = '#d62728'
    ax1.set_xlabel('Époque (Epoch)', fontsize=12)
    ax1.set_ylabel('Validation Loss', color=color1, fontsize=12)
    line1, = ax1.plot(epochs, val_loss, color=color1, marker='o', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-axis: Recall@1 (Blue/Green colors)
    ax2 = ax1.twinx()  
    color2 = '#1f77b4'
    color3 = '#2ca02c'
    ax2.set_ylabel('Recall@1', color='black', fontsize=12)  
    line2, = ax2.plot(epochs, i2t_r1, color=color2, linestyle='-', marker='s', label='Image-to-Text R@1')
    line3, = ax2.plot(epochs, t2i_r1, color=color3, linestyle='--', marker='^', label='Text-to-Image R@1')
    ax2.tick_params(axis='y', labelcolor='black')

    # Highlight the best Epoch 23
    best_epoch_idx = 23 - 1 # Epoch 23 is at index 22
    if best_epoch_idx < len(epochs):
        ax2.axvline(x=23, color='gray', linestyle=':', linewidth=1.5)
        ax2.annotate('Meilleur Modèle (Epoch 23)', 
                     xy=(23, i2t_r1[best_epoch_idx]), 
                     xytext=(15, 0.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=11)

    fig.tight_layout()  
    plt.title("Performances de l'entraînement du Modèle Teacher (ViT-B-16)", fontsize=14, pad=15)

    # Combine legends
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.grid(True, alpha=0.3)
    out_file = 'teacher_training_plot.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot has been generated and saved to: {out_file}")

if __name__ == "__main__":
    main()
