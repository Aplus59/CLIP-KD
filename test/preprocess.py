import os
import json
import torch
import shutil
from datasets import load_from_disk

# =========================================================
# SET THIS PATH EXPLICITLY
# =========================================================
TEST_ROOT = r"C:\Users\bkhanh\Desktop\code\M2\pj_t\CLIP-KD\test"
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

    print("\n=== Preprocessing Flickr30k (1K retrieval test) ===")

    dataset = load_from_disk(dataset_dir)

    if "test" not in dataset:
        raise ValueError("[Flickr30k] Expected a 'test' split in saved dataset.")

    test_split = dataset["test"]
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    captions_dict = {}
    keys = []

    for item in test_split:
        filename = os.path.basename(item["filename"])
        key = int(os.path.splitext(filename)[0])

        captions = item["caption"]
        if isinstance(captions, str):
            captions = [captions]

        captions = [c.strip() for c in captions if isinstance(c, str) and c.strip()]
        if not captions:
            continue

        captions_dict[key] = captions
        keys.append(key)

        img_path = os.path.join(images_dir, f"{key}.jpg")
        if not os.path.exists(img_path):
            item["image"].save(img_path)

    keys = sorted(set(keys))
    captions_dict = {k: captions_dict[k] for k in keys}

    print(f"[Flickr30k] Parsed test images: {len(keys)}")
    print(f"[Flickr30k] Done. Total test images: {len(keys)}")

    save_captions_pt(captions_dict, dataset_dir)
    save_img_keys_tsv(keys, dataset_dir)
    
def preprocess_coco(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"[MSCOCO] Missing directory: {dataset_dir}")
        return

    print("\n=== Preprocessing MSCOCO (Karpathy test 5k) ===")

    # ---------------------------------------------------------
    # Fix image folder structure: cocoir/val2014 -> cocoir/images/val2014
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Find split file
    # ---------------------------------------------------------
    split_candidates = [
        os.path.join(dataset_dir, "coco_karpathy_test.json"),
        os.path.join(dataset_dir, "dataset_coco.json"),
        os.path.join(dataset_dir, "karpathy_test.json"),
        os.path.join(dataset_dir, "annotations", "coco_karpathy_test.json"),
        os.path.join(dataset_dir, "annotations", "dataset_coco.json"),
        os.path.join(dataset_dir, "annotations", "karpathy_test.json"),
    ]

    split_path = None
    for p in split_candidates:
        if os.path.exists(p):
            split_path = p
            break

    if split_path is None:
        raise FileNotFoundError(
            "[MSCOCO] Missing split file. Expected one of:\n"
            "  - coco_karpathy_test.json\n"
            "  - dataset_coco.json\n"
            "  - karpathy_test.json"
        )

    print(f"[MSCOCO] Using split file: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    captions_dict = {}
    keys = []

    # ---------------------------------------------------------
    # Helper: extract image_id from filename/path
    # e.g. val2014/COCO_val2014_000000391895.jpg -> 391895
    # ---------------------------------------------------------
    def extract_coco_id_from_path(path_str):
        base = os.path.basename(path_str)
        stem = os.path.splitext(base)[0]
        # COCO_val2014_000000391895 -> 391895
        return int(stem.split("_")[-1])

    # ---------------------------------------------------------
    # Case A: Karpathy original dataset_coco.json
    # ---------------------------------------------------------
    if isinstance(split_data, dict) and "images" in split_data and isinstance(split_data["images"], list):
        # original Karpathy dataset_coco.json format
        # keep only split == test
        found_original = False
        for item in split_data["images"]:
            if "split" in item:
                found_original = True
                if item.get("split") != "test":
                    continue

                if "cocoid" in item:
                    key = int(item["cocoid"])
                elif "filename" in item:
                    key = extract_coco_id_from_path(item["filename"])
                elif "image" in item:
                    key = extract_coco_id_from_path(item["image"])
                else:
                    continue

                caps = []
                for s in item.get("sentences", []):
                    if isinstance(s, dict):
                        if "raw" in s:
                            caps.append(s["raw"])
                        elif "caption" in s:
                            caps.append(s["caption"])
                    elif isinstance(s, str):
                        caps.append(s)

                caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if not caps:
                    continue

                captions_dict[key] = caps
                keys.append(key)

        if not found_original:
            # ---------------------------------------------------------
            # Case B: some jsons use {"images": [...]} but entries are flat ann records
            # ---------------------------------------------------------
            entries = split_data["images"]
            tmp_caps = {}

            for item in entries:
                key = None
                if "image_id" in item:
                    key = int(item["image_id"])
                elif "cocoid" in item:
                    key = int(item["cocoid"])
                elif "filename" in item:
                    key = extract_coco_id_from_path(item["filename"])
                elif "image" in item:
                    key = extract_coco_id_from_path(item["image"])

                if key is None:
                    continue

                caps = []
                if "captions" in item and isinstance(item["captions"], list):
                    caps.extend(item["captions"])
                elif "caption" in item:
                    caps.append(item["caption"])
                elif "sentences" in item:
                    for s in item["sentences"]:
                        if isinstance(s, dict):
                            if "raw" in s:
                                caps.append(s["raw"])
                            elif "caption" in s:
                                caps.append(s["caption"])
                        elif isinstance(s, str):
                            caps.append(s)

                caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if not caps:
                    continue

                tmp_caps.setdefault(key, []).extend(caps)

            for key, caps in tmp_caps.items():
                captions_dict[key] = caps
                keys.append(key)

    # ---------------------------------------------------------
    # Case C: coco_karpathy_test.json / list of annotations
    # Each record often looks like:
    # {"image": "val2014/COCO_val2014_000000391895.jpg", "caption": "...", ...}
    # ---------------------------------------------------------
    elif isinstance(split_data, list):
        tmp_caps = {}

        for item in split_data:
            if not isinstance(item, dict):
                continue

            key = None
            if "image_id" in item:
                key = int(item["image_id"])
            elif "cocoid" in item:
                key = int(item["cocoid"])
            elif "filename" in item:
                key = extract_coco_id_from_path(item["filename"])
            elif "image" in item:
                key = extract_coco_id_from_path(item["image"])

            if key is None:
                continue

            caps = []

            # caption có thể là string hoặc list[string]
            if "caption" in item:
                if isinstance(item["caption"], str):
                    caps.append(item["caption"])
                elif isinstance(item["caption"], list):
                    caps.extend([c for c in item["caption"] if isinstance(c, str)])

            if "captions" in item and isinstance(item["captions"], list):
                caps.extend([c for c in item["captions"] if isinstance(c, str)])

            if "sentences" in item:
                for s in item["sentences"]:
                    if isinstance(s, dict):
                        if "raw" in s:
                            caps.append(s["raw"])
                        elif "caption" in s:
                            if isinstance(s["caption"], str):
                                caps.append(s["caption"])
                            elif isinstance(s["caption"], list):
                                caps.extend([c for c in s["caption"] if isinstance(c, str)])
                    elif isinstance(s, str):
                        caps.append(s)

            caps = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
            if not caps:
                continue

            tmp_caps.setdefault(key, []).extend(caps)

        for key, caps in tmp_caps.items():
            captions_dict[key] = caps
            keys.append(key)

    else:
        raise ValueError(f"[MSCOCO] Unsupported split format in {split_path}")

    # ---------------------------------------------------------
    # Sort keys and deduplicate captions
    # ---------------------------------------------------------
    keys = sorted(set(int(k) for k in keys))

    cleaned = {}
    for k in keys:
        if k not in captions_dict:
            continue
        seen = set()
        uniq_caps = []
        for c in captions_dict[k]:
            c = c.strip()
            if c and c not in seen:
                uniq_caps.append(c)
                seen.add(c)
        if uniq_caps:
            cleaned[k] = uniq_caps

    captions_dict = cleaned
    keys = sorted(captions_dict.keys())

    # ---------------------------------------------------------
    # Check size
    # ---------------------------------------------------------
    if len(keys) != 5000:
        print(f"[MSCOCO] Warning: expected 5000 test images, got {len(keys)}")
    else:
        print("[MSCOCO] Detected correct Karpathy test size: 5000 images")

    # ---------------------------------------------------------
    # Check missing images
    # ---------------------------------------------------------
    missing_count = 0
    for key in keys:
        img_name = f"COCO_val2014_{key:012d}.jpg"
        img_path = os.path.join(val2014_new, img_name)
        if not os.path.exists(img_path):
            print(f"[MSCOCO] Warning: missing image {img_path}")
            missing_count += 1

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    save_captions_pt(captions_dict, dataset_dir)
    save_img_keys_tsv(keys, dataset_dir)

    print(
        f"[MSCOCO] Done. Total Karpathy test images: {len(keys)} | "
        f"Missing images: {missing_count}"
    )

def list_subdirs(root_dir):
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])


def find_deepest_class_root(root_dir, expected_name=None):
    """
    Tìm thư mục class root thực sự.
    Nếu có expected_name và tồn tại thì ưu tiên dùng.
    Nếu root chỉ có 1 thư mục con, chui tiếp vào trong.
    """
    if not os.path.exists(root_dir):
        return None

    if expected_name:
        candidate = os.path.join(root_dir, expected_name)
        if os.path.isdir(candidate):
            return candidate

    current = root_dir
    while True:
        subdirs = list_subdirs(current)
        if len(subdirs) != 1:
            return current
        next_dir = os.path.join(current, subdirs[0])
        current = next_dir


def rename_imagenet_v2_numeric_folders(class_root):
    """
    Đổi tên folder ImageNet-V2:
    0 -> 000, 1 -> 001, ..., 999 -> 999
    để tránh lỗi ImageFolder sort theo chữ cái.
    """
    if not os.path.exists(class_root):
        print(f"[ImageNet-V2] Missing class root: {class_root}")
        return

    subdirs = list_subdirs(class_root)
    numeric_dirs = [d for d in subdirs if d.isdigit()]

    if not numeric_dirs:
        print("[ImageNet-V2] No numeric class folders found. Skip renaming.")
        return

    if all(len(d) == 3 for d in numeric_dirs):
        print("[ImageNet-V2] Folder names already zero-padded. Skip renaming.")
        return

    print(f"[ImageNet-V2] Renaming {len(numeric_dirs)} numeric class folders to zero-padded format...")

    # Bước 1: đổi sang tên tạm để tránh đè nhau
    temp_pairs = []
    for name in numeric_dirs:
        old_path = os.path.join(class_root, name)
        temp_name = f"tmp_{name}"
        temp_path = os.path.join(class_root, temp_name)
        os.rename(old_path, temp_path)
        temp_pairs.append((temp_name, f"{int(name):03d}"))

    # Bước 2: đổi sang tên cuối
    for temp_name, final_name in temp_pairs:
        temp_path = os.path.join(class_root, temp_name)
        final_path = os.path.join(class_root, final_name)
        os.rename(temp_path, final_path)
        print(f"[ImageNet-V2] {temp_name} -> {final_name}")

    print("[ImageNet-V2] Renaming completed.")


def inspect_imagenet_folder(root_dir, name, expected_inner_dir=None, fix_v2_numeric=False):
    print(f"\n=== Checking {name} ===")
    if not os.path.exists(root_dir):
        print(f"[{name}] Missing: {root_dir}")
        return None

    class_root = find_deepest_class_root(root_dir, expected_inner_dir)
    print(f"[{name}] Detected class root: {class_root}")

    if fix_v2_numeric:
        rename_imagenet_v2_numeric_folders(class_root)

    subdirs = list_subdirs(class_root)
    print(f"[{name}] Found {len(subdirs)} class folders in {class_root}")

    if len(subdirs) > 0:
        print(f"[{name}] First 10 folders: {subdirs[:10]}")

    if len(subdirs) != 1000 and name == "ImageNet-V2":
        print(f"[{name}] Warning: expected about 1000 class folders, got {len(subdirs)}")
    elif len(subdirs) != 200 and name == "ImageNet-R":
        print(f"[{name}] Note: ImageNet-R usually has 200 classes, got {len(subdirs)}")
    elif len(subdirs) != 1000 and name == "ImageNet-Sketch":
        print(f"[{name}] Warning: expected about 1000 class folders, got {len(subdirs)}")

    print(f"[{name}] No .pt/.tsv preprocess file needed for CLIP-KD ImageFolder datasets.")
    return class_root


if __name__ == "__main__":
    print(f"Using TEST_ROOT = {TEST_ROOT}")

    preprocess_flickr30k(flickr_dir)
    # preprocess_coco(coco_dir)

    v2_root = inspect_imagenet_folder(
        imagenet_v2_dir,
        "ImageNet-V2",
        expected_inner_dir="imagenetv2-matched-frequency-format-val",
        fix_v2_numeric=True,
    )

    r_root = inspect_imagenet_folder(
        imagenet_r_dir,
        "ImageNet-R",
        expected_inner_dir="imagenet-r",
        fix_v2_numeric=False,
    )

    sketch_root = inspect_imagenet_folder(
        imagenet_sketch_dir,
        "ImageNet-Sketch",
        expected_inner_dir="sketch",
        fix_v2_numeric=False,
    )

    print("\nDone.")
    print("\nRecommended eval paths:")
    print(f"  --imagenet-v2 {v2_root}")
    print(f"  --imagenet-r {r_root}")
    print(f"  --imagenet-sketch {sketch_root}")