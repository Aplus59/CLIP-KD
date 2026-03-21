import os
import requests
import zipfile
import tarfile
from datasets import load_dataset

# Lấy thư mục hiện tại (test)
base_dir = os.getcwd()

# Đường dẫn cho từng dataset
flickr_dir = os.path.join(base_dir, 'flickrir')
coco_dir = os.path.join(base_dir, 'cocoir')
imagenet_v2_dir = os.path.join(base_dir, 'imagenet-v2')
imagenet_r_dir = os.path.join(base_dir, 'imagenet-r')
imagenet_sketch_dir = os.path.join(base_dir, 'imagenet-sketch')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(flickr_dir, exist_ok=True)
os.makedirs(coco_dir, exist_ok=True)
os.makedirs(imagenet_v2_dir, exist_ok=True)
os.makedirs(imagenet_r_dir, exist_ok=True)
os.makedirs(imagenet_sketch_dir, exist_ok=True)

def is_directory_not_empty(directory):
    if os.path.exists(directory):
        return len(os.listdir(directory)) > 0
    return False

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {dest_path}")

def download_and_extract(url, dest_dir, is_tar=False):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif is_tar or filename.endswith('.tar') or filename.endswith('.tar.gz'):
        with tarfile.open(filepath, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)

    os.remove(filepath)
    print(f"Extracted to: {dest_dir}")

def get_hf_parquet_urls(dataset_name):
    api = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
    r = requests.get(api, timeout=60)
    r.raise_for_status()
    data = r.json()

    parquet_files = []
    for item in data.get("parquet_files", []):
        split = item.get("split")
        url = item.get("url")
        if split == "test" and url:
            parquet_files.append(url)

    if not parquet_files:
        raise RuntimeError(
            f"No parquet URLs found for test split of {dataset_name}. "
            f"Response keys: {list(data.keys())}"
        )

    return parquet_files

# 1. Flickr30k 1K retrieval test from Parquet
# Nếu muốn tải lại sạch, hãy xóa thư mục flickrir trước khi chạy.
flickr_marker = os.path.join(flickr_dir, "dataset_info.json")

if not os.path.exists(flickr_marker):
    print("Downloading Flickr30k 1K retrieval test via Parquet...")
    flickr_dataset_name = "nlphuji/flickr_1k_test_image_text_retrieval"
    parquet_urls = get_hf_parquet_urls(flickr_dataset_name)
    print(f"Found {len(parquet_urls)} parquet file(s) for Flickr test split")

    dataset = load_dataset(
        "parquet",
        data_files={"test": parquet_urls},
    )
    dataset.save_to_disk(flickr_dir)
    print(f"Flickr30k 1K retrieval test saved to {flickr_dir}")
else:
    print(f"Flickr dataset already exists in {flickr_dir}, skipping download.")

# 2. MSCOCO 2014 Val
if not is_directory_not_empty(coco_dir):
    print("Downloading MSCOCO 2014 Val images...")
    download_and_extract("http://images.cocodataset.org/zips/val2014.zip", coco_dir)

    print("Downloading MSCOCO 2014 annotations...")
    download_and_extract("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", coco_dir)

    print(f"MSCOCO saved to {coco_dir}")
else:
    print(f"MSCOCO already exists in {coco_dir}, skipping download.")

# 2.1 COCO Karpathy test split
karpathy_test_path = os.path.join(coco_dir, "coco_karpathy_test.json")
karpathy_test_url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"

if not os.path.exists(karpathy_test_path):
    print("Downloading COCO Karpathy test split...")
    download_file(karpathy_test_url, karpathy_test_path)
    print(f"Karpathy test split saved to {karpathy_test_path}")
else:
    print(f"Karpathy test split already exists at {karpathy_test_path}, skipping download.")

# 3. ImageNet Variants
# 3.1 ImageNet-V2
if not is_directory_not_empty(imagenet_v2_dir):
    print("Downloading ImageNet-V2...")
    download_and_extract(
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        imagenet_v2_dir,
        is_tar=True
    )
    print(f"ImageNet-V2 saved to {imagenet_v2_dir}")
else:
    print(f"ImageNet-V2 already exists in {imagenet_v2_dir}, skipping download.")

# 3.2 ImageNet-R
if not is_directory_not_empty(imagenet_r_dir):
    print("Downloading ImageNet-R...")
    download_and_extract(
        "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
        imagenet_r_dir,
        is_tar=True
    )
    print(f"ImageNet-R saved to {imagenet_r_dir}")
else:
    print(f"ImageNet-R already exists in {imagenet_r_dir}, skipping download.")

# 3.3 ImageNet-Sketch
try:
    import gdown
except ImportError:
    print("gdown not installed. Please run 'pip install gdown' and try again.")
    raise

if not is_directory_not_empty(imagenet_sketch_dir):
    print("Downloading ImageNet-Sketch using gdown...")
    url = "https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"
    filename = 'sketch.zip'
    filepath = os.path.join(imagenet_sketch_dir, filename)
    gdown.download(url, filepath, quiet=False)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(imagenet_sketch_dir)

    os.remove(filepath)
    print(f"ImageNet-Sketch saved to {imagenet_sketch_dir}")
else:
    print(f"ImageNet-Sketch already exists in {imagenet_sketch_dir}, skipping download.")