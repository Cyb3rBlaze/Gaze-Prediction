import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image, ImageOps

from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt

import pickle


# custom dataset used to load gaze data
class Dataset(Dataset):
    def __init__(self, directory):
        self.directory = directory

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

        print("Loading dataset sample names...")

        train_dir = self.directory

        # Create lists will all training and test image file names, sorted
        self.train_folder_list = os.listdir(train_dir)

        self.train_img_list = []
        self.train_targets_list = []

        for folder in self.train_folder_list:
            self.train_img_list += [train_dir + "/" + folder + "/" + img for img in os.listdir(train_dir + "/" + folder) if img[len(img)-3:] == "png"]
            self.train_targets_list += [train_dir + "/" + folder + "/" + target for target in os.listdir(train_dir + "/" + folder) if target[len(target)-3:] == "pkl"]
        
        self.train_img_list.sort()
        self.train_targets_list.sort()

        print('Training images: ' + str(len(self.train_img_list)))


    def __len__(self):
        return len(self.train_img_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.transform(Image.open(self.train_img_list[idx]).convert('RGB'))
        with open(self.train_targets_list[idx], 'rb') as f:
            target = pickle.load(f)

        return (image, torch.tensor(target["look_vec"], dtype=torch.float32)[:2], torch.tensor(target["head_pose"]))