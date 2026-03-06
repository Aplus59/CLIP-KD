import os
import requests
import zipfile
import tarfile
from datasets import load_dataset  # Cần pip install datasets nếu chưa có
import warnings

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

# Hàm kiểm tra thư mục có file không
def is_directory_not_empty(directory):
    if os.path.exists(directory):
        return len(os.listdir(directory)) > 0
    return False

# Hàm tải và extract zip/tar
def download_and_extract(url, dest_dir, is_tar=False):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        elif is_tar or filename.endswith('.tar') or filename.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(dest_dir)
        os.remove(filepath)
    else:
        print(f"Error downloading {url}: {response.status_code}")

# 1. Flickr30k
if not is_directory_not_empty(flickr_dir):
    print("Downloading Flickr30k...")
    dataset = load_dataset("lmms-lab/flickr30k")
    dataset.save_to_disk(flickr_dir)
    print(f"Flickr30k saved to {flickr_dir}")
else:
    print(f"Flickr30k already exists in {flickr_dir}, skipping download.")

# 2. MSCOCO 2014 Val
if not is_directory_not_empty(coco_dir):
    print("Downloading MSCOCO 2014 Val images...")
    download_and_extract("http://images.cocodataset.org/zips/val2014.zip", coco_dir)
    print("Downloading MSCOCO 2014 annotations...")
    download_and_extract("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", coco_dir)
    print(f"MSCOCO saved to {coco_dir}")
else:
    print(f"MSCOCO already exists in {coco_dir}, skipping download.")

# 3. ImageNet Variants
# 3.1 ImageNet-V2
if not is_directory_not_empty(imagenet_v2_dir):
    print("Downloading ImageNet-V2...")
    download_and_extract("https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz", imagenet_v2_dir, is_tar=True)
    print(f"ImageNet-V2 saved to {imagenet_v2_dir}")
else:
    print(f"ImageNet-V2 already exists in {imagenet_v2_dir}, skipping download.")

# 3.2 ImageNet-R
if not is_directory_not_empty(imagenet_r_dir):
    print("Downloading ImageNet-R...")
    download_and_extract("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", imagenet_r_dir, is_tar=True)
    print(f"ImageNet-R saved to {imagenet_r_dir}")
else:
    print(f"ImageNet-R already exists in {imagenet_r_dir}, skipping download.")

# 3.3 ImageNet-Sketch (sử dụng gdown để tải từ Google Drive - cần pip install gdown)
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