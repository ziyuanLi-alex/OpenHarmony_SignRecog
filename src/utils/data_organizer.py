import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

def organize_data():
    source_dir = ROOT_DIR / "dataset/raw/train/images"
    rgb_dir = ROOT_DIR / "dataset/raw/train/rgb"
    depth_dir = ROOT_DIR / "dataset/raw/train/depth"
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    for file in os.listdir(source_dir):
        if file.endswith(('.jpg', '.png')):
            src_path = source_dir / file
            if 'rgb' in file.lower():
                dst_path = rgb_dir / file
            elif 'depth' in file.lower():
                dst_path = depth_dir / file
            shutil.copy(str(src_path), str(dst_path))

def create_data_yaml():
    yaml_path = ROOT_DIR / "dataset/raw/data.yaml"
    
    # 移除J和Z，保留24个字母
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    yaml_content = f"""path: {ROOT_DIR / "dataset/raw"}  # dataset root dir
train: train/images  # train images
val: valid/images  # val images

nc: 24  # number of classes (excluding J and Z)
names: {letters}  # class names
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    organize_data()
    create_data_yaml()