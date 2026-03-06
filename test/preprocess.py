import os
import json
import torch
import pandas as pd
from PIL import Image  # Để check ảnh tồn tại và extract
import shutil  # Để tạo folder nếu cần

# Đường dẫn base
base_dir = os.getcwd()  # Thư mục test/

# Hàm chung để tạo test_captions.pt (dict {key: [captions]})
def save_captions_pt(captions_dict, dest_dir):
    torch.save(captions_dict, os.path.join(dest_dir, 'test_captions.pt'))
    print(f"Saved test_captions.pt to {dest_dir}")

# Hàm tạo test_img_keys.tsv (danh sách keys, mỗi dòng một key)
def save_img_keys_tsv(keys, dest_dir):
    with open(os.path.join(dest_dir, 'test_img_keys.tsv'), 'w') as f:
        for key in keys:
            f.write(f"{key}\n")
    print(f"Saved test_img_keys.tsv to {dest_dir}")

# 1. Preprocess Flickr30k
flickr_dir = os.path.join(base_dir, 'flickrir')
flickr_images_dir = os.path.join(flickr_dir, 'images')  # Tạo folder images nếu chưa có
os.makedirs(flickr_images_dir, exist_ok=True)

if os.path.exists(flickr_dir):
    from datasets import load_from_disk
    dataset = load_from_disk(flickr_dir)
    test_split = dataset['test']  # Flickr30k có 'test'
    captions_dict = {}
    keys = []
    for item in test_split:
        filename = item['filename']  # e.g., "1000092795.jpg"
        key = filename.replace('.jpg', '')  # e.g., "1000092795"
        captions = item['caption']  # Already a list of 5 captions
        captions_dict[key] = captions
        keys.append(key)
        
        # Extract image to images/ if not exists
        img_path = os.path.join(flickr_images_dir, filename)
        if not os.path.exists(img_path):
            image = item['image']  # PIL.Image
            image.save(img_path)
            print(f"Saved image {filename} to {flickr_images_dir}")
        else:
            print(f"Image {filename} already exists.")
    
    save_captions_pt(captions_dict, flickr_dir)
    save_img_keys_tsv(keys, flickr_dir)

# 2. Preprocess MSCOCO (từ annotations_trainval2014.zip và val2014.zip)
coco_dir = os.path.join(base_dir, 'cocoir')
coco_images_dir = os.path.join(coco_dir, 'val2014')  # Sau unzip, move images to here if needed
coco_annotations_dir = os.path.join(coco_dir, 'annotations')
coco_annotations = os.path.join(coco_annotations_dir, 'captions_val2014.json')
if os.path.exists(coco_annotations):
    with open(coco_annotations, 'r') as f:
        data = json.load(f)
    captions_dict = {}
    keys = []
    for ann in data['annotations']:
        key = ann['image_id']  # integer
        caption = ann['caption']
        if key not in captions_dict:
            captions_dict[key] = []
            keys.append(key)
        captions_dict[key].append(caption)
    # Check ảnh: val2014/COCO_val2014_{key:012d}.jpg
    for key in keys:
        img_path = os.path.join(coco_images_dir, f"COCO_val2014_{key:012d}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: Missing image {img_path}")
    save_captions_pt(captions_dict, coco_dir)
    save_img_keys_tsv(keys, coco_dir)

# ImageNet variants (không cần preprocess, dùng folder class trực tiếp cho eval classification)
# Ví dụ: imagenet-v2/imagenetv2-matched-frequency-format-val/ chứa thư mục 0-999 cho classes