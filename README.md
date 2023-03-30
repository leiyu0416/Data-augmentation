# Data augmentation


C.

## Approach

![Data augmentation](picture.jpg)



## Usage

Here we apply the environment required by clip
First, [install PyTorch 1.8.1](https://pytorch.org/get-started/locally/) (or later) and torchvision. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

## More Examples

### Single image data expansion



```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30), expand=False),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    transforms.ToTensor(),
])

# 加载图片
image = Image.open('picture.jpg')

# 对图片进行增广
augmented_image = transform(image)

# 显示增广后的图片
import matplotlib.pyplot as plt
plt.imshow(augmented_image.permute(1, 2, 0))
plt.show()
```



### Processing data augmentation for training sets

The following example achieves horizontal flipping of image data with a probability of 50%; Image rotation, small angle; Image data cutting, with a cut size of 256 and a probability of 50%.

```python
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
```

The processed image here will be saved in 'train_result'

