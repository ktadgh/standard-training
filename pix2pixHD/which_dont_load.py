import time
import os
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.networks import DistillLoss
from thop import profile

import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
from aim import Run, Image

import cv2

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import copy
import lpips

import sys
import subprocess
import os
import numpy as np
import torchvision
from torcheval.metrics import PeakSignalNoiseRatio
import lpips
import torch
import re
from tqdm import tqdm


opt = TrainOptions().parse()
opt.serial_batches = True


iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)


print('#training images = %d' % dataset_size)



# Example of removing specific arguments
filtered_args = []
skip_next = False

color_min = 1e15
depth_min = 1e15
normal_min = 1e15

normal_max = -1e-15
depth_max = -1e15
color_max = -1e15

j = 0
for i, data in tqdm(enumerate(dataset)):
    j += 1
    # raise ValueError(data)
    # data should have the pattern 
    color_input = data['label'][:,:3,:,:]
    depth = data['label'][:,3:4,:,:]
    normal = data['label'][:,4:,:,:]

    if color_input.min() < color_min:
        color_min = color_input.min()
    if color_input.max() > color_max:
        color_max = color_input.max()

    if depth.min() < depth_min:
        depth_min = depth.min()
    if depth.max() > depth_max:
        depth_max = depth.max()

    if normal.min() < normal_min:
        normal_min = normal.min()
    if normal.max() > normal_max:
        normal_max = normal.max()

    if j > 10:
        break
print(f'Normal min = {normal_min}, Normal max = {normal_max}')
print(f'Depth min = {depth_min}, Depth max = {depth_max}')
print(f'Color min = {color_min}, Color max = {color_max}')