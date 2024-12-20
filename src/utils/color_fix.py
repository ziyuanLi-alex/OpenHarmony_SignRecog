import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

def fix_color(input_dir: str, output_dir: str = None):
    """
    修正图片的颜色问题(RGB->BGR)
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录，默认为None则覆盖原文件
    """
    import os
    import cv2
    from pathlib import Path
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有rgb图片
    for img_path in Path(input_dir).glob("*_rgb.jpg"):
        # 读取图片
        img = cv2.imread(str(img_path))
        
        # RGB转回BGR
        img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 确定保存路径
        save_path = os.path.join(output_dir, img_path.name) if output_dir else img_path
        
        # 保存图片
        cv2.imwrite(str(save_path), img_fixed)
        print(f"Processed: {img_path.name}")

# 使用示例:
# fix_color("./train/rgb", "./train/rgb_fixed")  # 保存到新目录
# fix_color("./train/rgb")  # 覆盖原文件

if __name__ == "__main__":
    # fix_color(ROOT_DIR / "dataset/rgb/images/test")
    # fix_color(ROOT_DIR / "dataset/rgb/images/train")
    # fix_color(ROOT_DIR / "dataset/rgb/images/val")
    fix_color(ROOT_DIR / "dataset/raw/train/images")
              