import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from util.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class IRSeg(data.Dataset):

    def __init__(self, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = './dataset'  ###  your dataset path  ###
        self.n_classes = 9

        scale_range = [0.5, 2.0]
        crop_size =[480, 640]

        self.aug = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            #RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])


        self.mode = mode
        self.do_aug = do_aug
        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()


        image = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_rgb.png'))
        thermal = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_th.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        sample = {
            'image': image,
            'thermal': thermal,
            'label': label
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:  
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.dp_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (64, 0, 128),  # car
            (64, 64, 0),  # person
            (0, 128, 192),  # bike
            (0, 0, 192),  # curve
            (128, 128, 0),  # car_stop
            (64, 64, 128),  # guardrail
            (192, 128, 128),  # color_cone
            (192, 64, 0),  # bump
        ]


