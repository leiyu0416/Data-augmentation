import os
import torch
import torchvision.transforms as transforms
from PIL import Image
# 定义数据增强函数
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30), expand=False),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.8, 1.2)),])
# 加载文件夹
image_folder = 'F:\\classification-basic-sample-master\\data\\train\\train'
# 获取文件夹中的所有图片文件名
image_filenames = os.listdir(image_folder)
# 定义保存增强后图片的文件夹路径
output_folder = 'F:\\classification-basic-sample-master\\train_result'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 对每张图片进行增强
for image_filename in image_filenames:
    # 加载图片
    image_path = os.path.join(image_folder, image_filename)
    image = Image.open(image_path)
    # 对图片进行增强
    augmented_image = transform(image)
    # 保存增强后的图片
    output_path = os.path.join(output_folder, image_filename)
    augmented_image.save(output_path)