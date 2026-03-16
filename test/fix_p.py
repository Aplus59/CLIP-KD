import os
import shutil

base_dir = os.getcwd()  # test folder
coco_dir = os.path.join(base_dir, 'cocoir')
val2014_old = os.path.join(coco_dir, 'val2014')
images_dir = os.path.join(coco_dir, 'images')
val2014_new = os.path.join(images_dir, 'val2014')

if os.path.exists(val2014_old) and not os.path.exists(val2014_new):
    os.makedirs(images_dir, exist_ok=True)
    shutil.move(val2014_old, images_dir)
    print(f"Moved val2014 to {images_dir}/val2014")
else:
    print("Structure already correct or val2014 missing.")