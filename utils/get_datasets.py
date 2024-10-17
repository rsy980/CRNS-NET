import os
import random
import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label


# 输入增强

def random_brightness(image, brightness_factor=0.5):
    # 随机生成一个亮度增益因子，可以根据需要调整范围
    brightness_factor = np.random.uniform(1 - brightness_factor, 1 + brightness_factor)

    # 将图像的每个通道乘以亮度增益因子
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    return image


def random_contrast(image, contrast_factor=0.25):
# 随机生成一个对比度增益因子，可以根据需要调整范围
    contrast_factor = np.random.uniform(1 - contrast_factor, 1 + contrast_factor)

    # 将图像减去均值以使其具有零均值
    mean_value = np.mean(image)
    image = image - mean_value

    # 将图像的每个通道乘以对比度增益因子
    image = np.clip(image * contrast_factor + mean_value, 0, 255).astype(np.uint8)

    return image


def random_saturation(image, saturation_factor=0.5):
    # 随机生成一个饱和度增益因子，可以根据需要调整范围
    saturation_factor = np.random.uniform(1 - saturation_factor, 1 + saturation_factor)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整饱和度通道
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    # 转换回BGR颜色空间
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image


def random_color(image, color_factor=0.5):
    # 随机生成一个颜色增益因子，可以根据需要调整范围
    color_factor = np.random.uniform(1 - color_factor, 1 + color_factor)

    # 将图像从BGR颜色空间转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 调整a和b通道
    lab[:, :, 1] = np.clip(lab[:, :, 1] * color_factor, 0, 255).astype(np.uint8)
    lab[:, :, 2] = np.clip(lab[:, :, 2] * color_factor, 0, 255).astype(np.uint8)

    # 转换回BGR颜色空间
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image




class RandomGenerator(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 形状增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        augmentation_type = random.choice(['brightness', 'contrast', 'saturation', 'color', 'none'])

        if augmentation_type == 'brightness':
            # 随机亮度增强
            image = random_brightness(image, brightness_factor=0.5)
        elif augmentation_type == 'contrast':
            # 随机对比度增强
            image = random_contrast(image, contrast_factor=0.25)
        elif augmentation_type == 'saturation':
            # 随机饱和度增强
            image = random_saturation(image, saturation_factor=0.5)
        elif augmentation_type == 'color':
            # 随机颜色增强
            image = random_color(image, color_factor=0.1)
        else:
            pass

        x, y,_ = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))   #.unsqueeze(0)
        image = image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class GetDatasets(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2,0,1)
            label = torch.from_numpy(label.astype(np.float32))


        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
