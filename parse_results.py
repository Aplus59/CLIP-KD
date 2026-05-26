import glob
import re
import os

def parse_logs():
    log_files = glob.glob('logs/2026_05_26-*/out.log')
    if not log_files:
        print("Không tìm thấy file out.log nào trong logs/2026_05_26-*")
        return

    print(f"{'Run Name':<80} | {'Best Val Loss':<15} | {'Best I2T R@1':<12} | {'Best T2I R@1':<12}")
    print("-" * 130)

    for f in sorted(log_files):
        # Handle path separators on Windows and Linux
        parts = f.replace('\\', '/').split('/')
        run_name = parts[-2]
        
        best_val_loss = float('inf')
        best_i2t = 0.0
        best_t2i = 0.0
        
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                # Phù hợp với dòng: Eval Epoch: 1 image_to_text_R@1: 0.0889  image_to_text_R@5: 0.2000  ... val_loss: 3.1652
                matches = re.findall(
                    r"Eval Epoch:.*image_to_text_R@1:\s*([\d\.]+).*text_to_image_R@1:\s*([\d\.]+).*val_loss:\s*([\d\.]+)", 
                    content
                )
                for i2t, t2i, val_loss in matches:
                    val_loss = float(val_loss)
                    i2t = float(i2t)
                    t2i = float(t2i)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    if i2t > best_i2t:
                        best_i2t = i2t
                    if t2i > best_t2i:
                        best_t2i = t2i
        except Exception as e:
            print(f"Lỗi khi đọc file {f}: {e}")
            continue
            
        val_loss_str = f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
        i2t_str = f"{best_i2t:.4f}" if best_val_loss != float('inf') else "N/A"
        t2i_str = f"{best_t2i:.4f}" if best_val_loss != float('inf') else "N/A"
        
        print(f"{run_name:<80} | {val_loss_str:<15} | {i2t_str:<12} | {t2i_str:<12}")

if __name__ == '__main__':
    parse_logs()
