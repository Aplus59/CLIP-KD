import os
import json
import torch
import shutil
from datasets import load_from_disk

# =========================================================
# Base paths
# =========================================================
base_dir = os.getcwd()  # thư mục test/

flickr_dir = os.path.join(base_dir, "flickrir")
coco_dir = os.path.join(base_dir, "cocoir")
imagenet_v2_dir = os.path.join(base_dir, "imagenet-v2")
imagenet_r_dir = os.path.join(base_dir, "imagenet-r")
imagenet_sketch_dir = os.path.join(base_dir, "imagenet-sketch")


# =========================================================
# Common helpers
# =========================================================
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

def is_nonempty_dir(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


# =========================================================
# 1. Flickr30k
# CLIP-KD expects:
#   flickrir/
#     test_captions.pt
#     test_img_keys.tsv
#     images/<img_key>.jpg
# =========================================================
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
        filename = item["filename"]  # e.g. 1000092795.jpg
        key = os.path.splitext(filename)[0]  # "1000092795"
        captions = item["caption"]           # list of captions

        captions_dict[key] = captions
        keys.append(key)

        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            image = item["image"]  # PIL.Image
            image.save(img_path)

    save_captions_pt(captions_dict, dataset_dir)
    save_img_keys_tsv(keys, dataset_dir)
    print(f"[Flickr30k] Done. Total test images: {len(keys)}")


# =========================================================
# 2. MSCOCO
# CLIP-KD expects:
#   cocoir/
#     test_captions.pt
#     test_img_keys.tsv
#     images/val2014/COCO_val2014_xxxxxxxxxxxx.jpg
# =========================================================
def preprocess_coco(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"[MSCOCO] Missing directory: {dataset_dir}")
        return

    print("\n=== Preprocessing MSCOCO ===")

    # chuẩn hóa thư mục ảnh:
    # cocoir/val2014 -> cocoir/images/val2014
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
        key = ann["image_id"]   # int
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


# =========================================================
# 3. ImageNet-style datasets for CLIP-KD
# IMPORTANT:
# - Do NOT create test_labels.pt or test_img_keys.tsv here.
# - CLIP-KD reads these datasets directly with ImageFolder.
# - We only canonicalize/check folder structure.
# =========================================================
def count_class_folders(root_dir):
    if not os.path.isdir(root_dir):
        return 0
    return sum(
        1 for x in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, x))
    )

def ensure_imagenet_v2(dataset_dir):
    """
    Expected by CLIP-KD:
      --imagenet-v2=path/to/ImageNetV2-matched-frequency/
    Common extracted structure:
      imagenet-v2/
        imagenetv2-matched-frequency-format-val/
          0/
          1/
          ...
    We do not move by default; we only report the correct path to use.
    """
    print("\n=== Checking ImageNet-V2 ===")

    candidate_1 = os.path.join(dataset_dir, "imagenetv2-matched-frequency-format-val")
    candidate_2 = dataset_dir

    chosen = None
    if count_class_folders(candidate_1) > 0:
        chosen = candidate_1
    elif count_class_folders(candidate_2) > 0:
        chosen = candidate_2

    if chosen is None:
        print("[ImageNet-V2] Could not find class-folder root.")
        return None

    print(f"[ImageNet-V2] OK. Use this path for --imagenet-v2:\n  {chosen}")
    return chosen

def ensure_imagenet_r(dataset_dir):
    """
    Expected by CLIP-KD:
      --imagenet-r=path/to/imagenet-rendition/imagenet-r/
    Common extracted structure:
      imagenet-r/
        imagenet-r/
          <class_folder>/
    We keep class folders untouched because CLIP-KD uses ImageFolder
    and handles ImageNet-R in zero_shot.py via imagenet_r_indices.
    """
    print("\n=== Checking ImageNet-R ===")

    candidate_1 = os.path.join(dataset_dir, "imagenet-r")
    candidate_2 = dataset_dir

    chosen = None
    if count_class_folders(candidate_1) > 0:
        chosen = candidate_1
    elif count_class_folders(candidate_2) > 0:
        chosen = candidate_2

    if chosen is None:
        print("[ImageNet-R] Could not find class-folder root.")
        return None

    print(f"[ImageNet-R] OK. Use this path for --imagenet-r:\n  {chosen}")
    return chosen

def ensure_imagenet_sketch(dataset_dir):
    """
    Expected by CLIP-KD:
      --imagenet-sketch=path/to/imagenet-sketch/sketch/
    Common extracted structure:
      imagenet-sketch/
        sketch/
          <class_folder>/
    """
    print("\n=== Checking ImageNet-Sketch ===")

    candidate_1 = os.path.join(dataset_dir, "sketch")
    candidate_2 = os.path.join(dataset_dir, "imagenet-sketch")
    candidate_3 = dataset_dir

    chosen = None
    for candidate in [candidate_1, candidate_2, candidate_3]:
        if count_class_folders(candidate) > 0:
            chosen = candidate
            break

    if chosen is None:
        print("[ImageNet-Sketch] Could not find class-folder root.")
        return None

    print(f"[ImageNet-Sketch] OK. Use this path for --imagenet-sketch:\n  {chosen}")
    return chosen


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # Retrieval datasets: must preprocess
    preprocess_flickr30k(flickr_dir)
    preprocess_coco(coco_dir)

    # ImageNet-style datasets: only check/canonicalize path usage
    imagenet_v2_path = ensure_imagenet_v2(imagenet_v2_dir)
    imagenet_r_path = ensure_imagenet_r(imagenet_r_dir)
    imagenet_sketch_path = ensure_imagenet_sketch(imagenet_sketch_dir)

    print("\n=== Summary for CLIP-KD ===")
    print(f"Flickr30k root      : {flickr_dir}")
    print(f"MSCOCO root         : {coco_dir}")
    print(f"ImageNet-V2 arg     : {imagenet_v2_path}")
    print(f"ImageNet-R arg      : {imagenet_r_path}")
    print(f"ImageNet-Sketch arg : {imagenet_sketch_path}")
    print("\nDone.")