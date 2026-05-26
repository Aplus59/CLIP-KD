import os
import json
import glob

def get_best_results(log_dir):
    results_file = os.path.join(log_dir, "checkpoints", "results.jsonl")
    if not os.path.exists(results_file):
        return None

    best_epoch = -1
    best_score = -1
    best_data = None

    with open(results_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            i2t_r1 = data.get("image_to_text_R@1", 0)
            t2i_r1 = data.get("text_to_image_R@1", 0)
            avg_r1 = (i2t_r1 + t2i_r1) / 2.0
            
            if avg_r1 > best_score:
                best_score = avg_r1
                best_epoch = data.get("epoch", -1)
                best_data = data
                
    if best_score > 0 and best_data:
        return {
            "epoch": best_epoch,
            "val_loss": best_data.get("val_loss", 0),
            "i2t_r1": best_data.get("image_to_text_R@1", 0),
            "i2t_r5": best_data.get("image_to_text_R@5", 0),
            "i2t_r10": best_data.get("image_to_text_R@10", 0),
            "t2i_r1": best_data.get("text_to_image_R@1", 0),
            "t2i_r5": best_data.get("text_to_image_R@5", 0),
            "t2i_r10": best_data.get("text_to_image_R@10", 0),
            "avg_r1": best_score
        }
    return None

def main():
    methods = ["CrossKD", "ICL", "GD", "FD", "CKD", "AFD", "CLIP-KD"]
    results = {}

    for method in methods:
        # Find the latest log directory for each method
        tag = f"fathomnet-distill-{method}-student-vit-t"
        log_dirs = glob.glob(f"logs/*{tag}*")
        log_dirs.sort(key=os.path.getmtime, reverse=True)
        
        if log_dirs:
            best_res = get_best_results(log_dirs[0])
            if best_res:
                results[method] = best_res

    if not results:
        print("No method has completed saving result logs yet.")
        return

    # Sort in descending order of Avg R@1
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]["avg_r1"], reverse=True)

    print("-" * 125)
    print(f"{'Method':<10} | {'Epoch':<6} | {'Val Loss':<8} | {'I2T R@1':<8} | {'I2T R@5':<8} | {'I2T R@10':<8} | {'T2I R@1':<8} | {'T2I R@5':<8} | {'T2I R@10':<8} | {'Avg R@1':<8}")
    print("-" * 125)
    
    for method in sorted_methods:
        res = results[method]
        val_loss = f"{res['val_loss']:.4f}"
        i2t_1 = f"{res['i2t_r1']*100:.2f}%"
        i2t_5 = f"{res['i2t_r5']*100:.2f}%"
        i2t_10 = f"{res['i2t_r10']*100:.2f}%"
        t2i_1 = f"{res['t2i_r1']*100:.2f}%"
        t2i_5 = f"{res['t2i_r5']*100:.2f}%"
        t2i_10 = f"{res['t2i_r10']*100:.2f}%"
        avg_1 = f"{res['avg_r1']*100:.2f}%"
        
        print(f"{method:<10} | {res['epoch']:<6} | {val_loss:<8} | {i2t_1:<8} | {i2t_5:<8} | {i2t_10:<8} | {t2i_1:<8} | {t2i_5:<8} | {t2i_10:<8} | {avg_1:<8}")
    print("-" * 125)

if __name__ == "__main__":
    main()
