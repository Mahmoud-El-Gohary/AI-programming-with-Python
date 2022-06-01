import json
import torch
import PIL
import glob
import random
import argparse 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_transforms = {    
        'train' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),    
        'valid' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])           
    ]),    
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])           
    ])}


image_datasets = {    
    'train' : datasets.ImageFolder(train_dir, transform = data_transforms['train']),
    'valid' : datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
    'test'  : datasets.ImageFolder(test_dir,  transform = data_transforms['test'] ),   
}
