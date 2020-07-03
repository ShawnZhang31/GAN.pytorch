# Copyright shawnzhang31. All Rights Reserved
import argparse
import json
import os
import h5py
import imageio
import numpy as np

from models.utils.utils import printProgressBar
from models.utils.image_transform import pil_loader

def saveImage(path:str, image):
    """
    保存图像
    @params:
        path        - Required : 图像保存的路 (str)
        image       - Required : 图像数据
    """
    return imageio.imwrite(path, image)

def celebaSetup(inputPath:str, outputPath:str, pathConfig="config_celeba_cropped.json"):
    """
    celeba数据集配置
    @params:
        inputPath       - Required : 数据集的路径 (str)
        outputPath      - Required : 数据集的解压路径 (str)
        pathConfig      - Optional : 数据集的图像裁剪json配置文件 (json file)
    """
    imgList = [f for f in os.listdir(inputPath) if os.path.splitext(f)[1] == '.jpg']
    cx = 89
    cy = 121

    nImgs = len(imgList)

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    for index, item in enumerate(imgList):
        printProgressBar(index, nImgs)
        path = os.path.join(inputPath, item)
        img = np.array(pil_loader(path))
        # 对图像进行裁剪
        img = img[cy-64 : cy+64, cx - 64 : cx + 64]
        path = os.path.join(outputPath, item)
        saveImage(path, img)

    printProgressBar(nImgs, nImgs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试脚本")
    parser.add_argument('dataset_name', type=str,
                        choices=['celeba'],
                        help="dataset名称")
    parser.add_argument("dataset_path", type=str,
                        help="输入数据集的路径")
    parser.add_argument("-o", help="输出数据集的路径", 
                        type=str, dest="output_dataset")
    parser.add_argument("-r", action="store_true",
                        dest="fast_training",
                        help="为fast training储存一些resized的数据集。建议使用HD数据集的时候使用")
    parser.add_argument('-m', dest="model_type",
                        type=str, default='DCGAN',
                        choices=['DCGAN'],
                        help="模型类型，默认是DCGAN")
    args = parser.parse_args()

    config = {"pathDB": args.dataset_path}
    config["config"] = {}

    moveLastScale = False
    keepOriginalDataset = True

    if args.dataset_name in ['celeba', 'celeba_cropped']:
        maxSize = 128

    if args.dataset_name == 'celeba_cropped':
        if args.output_dataset is None:
            raise AttributeError(
                    "请输入celebaCropped转存的路径")
        
        print("裁剪数据集...")
        celebaSetup(args.dataset_path, args.output_dataset)
        config["pathDB"] = args.output_dataset
        moveLastScale = True
        
    