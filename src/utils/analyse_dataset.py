import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib 

matplotlib.rc('font', family='WenQuanYi Micro Hei')

def analyze_dataset():
   # 假设图片命名格式为: A_1_rgb.jpg, A_1_depth.png 等
   ROOT_DIR = Path(__file__).resolve().parents[2]
   image_dir = ROOT_DIR / "dataset/raw/train/images"
   
   # 用defaultdict来计数
   class_counts = defaultdict(int)
   rgb_counts = defaultdict(int)
   depth_counts = defaultdict(int)
   
   # 遍历图片
   for img_file in os.listdir(image_dir):
       if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
           continue
           
       # 从文件名获取类别
       class_name = img_file.split('_')[0].upper()
       
       # 排除J和Z
       if class_name in ['J', 'Z']:
           continue
           
       # 分别统计RGB和深度图
       if 'rgb' in img_file.lower():
           rgb_counts[class_name] += 1
       elif 'depth' in img_file.lower():
           depth_counts[class_name] += 1
           
       class_counts[class_name] += 1
   
   # 打印统计信息
   print("\n数据集统计:")
   print(f"总图片数: {sum(class_counts.values())}")
   print("\n每个类别的图片数量:")
   for class_name in sorted(class_counts.keys()):
       print(f"类别 {class_name}: {class_counts[class_name]} (RGB: {rgb_counts[class_name]}, Depth: {depth_counts[class_name]})")
   
   # 计算平均值和标准差来评估数据平衡性
   counts = list(class_counts.values())
   avg = sum(counts) / len(counts)
   std = (sum((x - avg) ** 2 for x in counts) / len(counts)) ** 0.5
   print(f"\n平均每类图片数: {avg:.2f}")
   print(f"标准差: {std:.2f}")
   print(f"变异系数: {(std/avg):.2f}")  # 变异系数 > 0.2 表示数据不平衡
   
   # 绘制分布图
   plt.figure(figsize=(15, 5))
   plt.bar(class_counts.keys(), class_counts.values())
   plt.title('手语字母数据集分布')
   plt.xlabel('字母类别')
   plt.ylabel('图片数量')
   plt.grid(True, alpha=0.3)
   
   # 添加平均值线
   plt.axhline(y=avg, color='r', linestyle='--', label=f'平均值: {avg:.1f}')
   plt.legend()
   
   # 保存图表
   plot_path = ROOT_DIR / "docs/dataset_distribution.png"
   plt.savefig(plot_path)
   plt.close()
   
   return class_counts, rgb_counts, depth_counts

if __name__ == "__main__":
   analyze_dataset()