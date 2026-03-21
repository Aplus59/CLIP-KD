import os
import torch

root = r"C:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\test\flickrir"

caps = torch.load(os.path.join(root, "test_captions.pt"))
with open(os.path.join(root, "test_img_keys.tsv"), "r", encoding="utf-8") as f:
    keys = [int(x.strip()) for x in f if x.strip()]

print("num keys:", len(keys))
print("num caption entries:", len(caps))

missing = []
for k in keys:
    img = os.path.join(root, "images", f"{k}.jpg")
    if not os.path.exists(img):
        missing.append(k)

print("num missing images:", len(missing))
if missing:
    print("first 20 missing:", missing[:20])
else:
    first_key = keys[0]
    print("first key:", first_key)
    print("captions of first key:", caps[first_key][:2])
    print("all images exist.")