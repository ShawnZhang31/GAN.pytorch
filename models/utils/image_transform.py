# Copyright shawnzhang31. All Rights Reserved
import torchvision.transforms as Transforms

import os
import random
import numpy as np
from PIL import Image

def pil_loader(path:str):
    """
    使用pillow加载图像
    @params:
        path    - Required : 图像的路径 (str)
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")

# pil_loader("./data/img_align_celeba/000001.jpg")