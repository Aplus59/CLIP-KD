import os
import json
import glob

def get_best_results(log_dir):
    results_file = os.path.join(log_dir, "checkpoints", "results.jsonl")
    if not os.path.exists(results_file):
        return None

    best_epoch = -1
    best_score = -1
    best_i2t = 0
    best_t2i = 0
    val_loss = 0

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
                best_i2t = i2t_r1
                best_t2i = t2i_r1
                val_loss = data.get("val_loss", 0)
                
    if best_score > 0:
        return {
            "epoch": best_epoch,
            "val_loss": val_loss,
            "i2t_r1": best_i2t,
            "t2i_r1": best_t2i,
            "avg_r1": best_score
        }
    return None

def main():
    methods = ["CrossKD", "ICL", "GD", "FD", "CKD", "AFD"]
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

    print("-" * 85)
    print(f"{'Méthode (Method)':<18} | {'Best Epoch':<10} | {'I2T R@1 (%)':<15} | {'T2I R@1 (%)':<15} | {'Avg R@1 (%)':<15}")
    print("-" * 85)
    
    for method in sorted_methods:
        res = results[method]
        print(f"{method:<18} | Epoch {res['epoch']:<4} | {res['i2t_r1']*100:>8.2f}%      | {res['t2i_r1']*100:>8.2f}%      | {res['avg_r1']*100:>8.2f}%")
    print("-" * 85)

if __name__ == "__main__":
    main()
