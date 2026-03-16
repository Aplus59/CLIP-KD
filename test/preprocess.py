import os
import json
import torch
import shutil
from datasets import load_from_disk

# =========================================================
# SET THIS PATH EXPLICITLY
# =========================================================
TEST_ROOT = "/content/drive/MyDrive/CLIP-KD/test"

flickr_dir = os.path.join(TEST_ROOT, "flickrir")
coco_dir = os.path.join(TEST_ROOT, "cocoir")
imagenet_v2_dir = os.path.join(TEST_ROOT, "imagenet-v2")
imagenet_r_dir = os.path.join(TEST_ROOT, "imagenet-r")
imagenet_sketch_dir = os.path.join(TEST_ROOT, "imagenet-sketch")


def save_captions_pt(captions_dict, dest_dir):
    save_path = os.path.join(dest_dir, "test_captions.pt")
    torch.save(captions_dict, save_path)
    print(f"Saved: {save_path}")

def save_img_keys_tsv(keys, dest_dir):
    save_path = os.path.join(dest_dir, "test_img_keys.tsv")
    with open(save_path, "w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}\n")
    print(f"Saved: {save_path}")


def preprocess_flickr30k(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"[Flickr30k] Missing directory: {dataset_dir}")
        return

    print("\n=== Preprocessing Flickr30k ===")
    dataset = load_from_disk(dataset_dir)

    if "test" not in dataset:
        raise ValueError("[Flickr30k] No 'test' split found in dataset.")

    test_split = dataset["test"]
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    captions_dict = {}
    keys = []

    for item in test_split:
        filename = item["filename"]
        key = os.path.splitext(filename)[0]   # string key for Flickr
        captions = item["caption"]

        captions_dict[key] = captions
        keys.append(key)

        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            item["image"].save(img_path)

    save_captions_pt(captions_dict, dataset_dir)

    # CLIP-KD RetrievalDataset đang ép img_keys thành int
    # nên với Flickr key phải là số
    numeric_keys = [int(k) for k in keys]
    save_img_keys_tsv(numeric_keys, dataset_dir)

    print(f"[Flickr30k] Done. Total test images: {len(keys)}")


def preprocess_coco(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"[MSCOCO] Missing directory: {dataset_dir}")
        return

    print("\n=== Preprocessing MSCOCO ===")

    val2014_old = os.path.join(dataset_dir, "val2014")
    images_dir = os.path.join(dataset_dir, "images")
    val2014_new = os.path.join(images_dir, "val2014")

    if os.path.exists(val2014_old) and not os.path.exists(val2014_new):
        os.makedirs(images_dir, exist_ok=True)
        shutil.move(val2014_old, val2014_new)
        print(f"[MSCOCO] Moved {val2014_old} -> {val2014_new}")
    elif os.path.exists(val2014_new):
        print("[MSCOCO] Image folder structure already correct.")
    else:
        print("[MSCOCO] Warning: val2014 folder not found.")

    anno_path = os.path.join(dataset_dir, "annotations", "captions_val2014.json")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"[MSCOCO] Missing annotation file: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions_dict = {}
    keys = []

    for ann in data["annotations"]:
        key = int(ann["image_id"])
        caption = ann["caption"]

        if key not in captions_dict:
            captions_dict[key] = []
            keys.append(key)

        captions_dict[key].append(caption)

    missing_count = 0
    for key in keys:
        img_name = f"COCO_val2014_{key:012d}.jpg"
        img_path = os.path.join(val2014_new, img_name)
        if not os.path.exists(img_path):
            print(f"[MSCOCO] Warning: missing image {img_path}")
            missing_count += 1

    save_captions_pt(captions_dict, dataset_dir)
    save_img_keys_tsv(keys, dataset_dir)

    print(f"[MSCOCO] Done. Total test images: {len(keys)} | Missing images: {missing_count}")


def check_imagenet_folder(root_dir, name):
    print(f"\n=== Checking {name} ===")
    if not os.path.exists(root_dir):
        print(f"[{name}] Missing: {root_dir}")
        return
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"[{name}] Found {len(subdirs)} subfolders in {root_dir}")
    print(f"[{name}] No preprocess file needed for CLIP-KD ImageFolder datasets.")


if __name__ == "__main__":
    print(f"Using TEST_ROOT = {TEST_ROOT}")

    preprocess_flickr30k(flickr_dir)
    preprocess_coco(coco_dir)

    check_imagenet_folder(imagenet_v2_dir, "ImageNet-V2")
    check_imagenet_folder(imagenet_r_dir, "ImageNet-R")
    check_imagenet_folder(imagenet_sketch_dir, "ImageNet-Sketch")

    print("\nDone.")