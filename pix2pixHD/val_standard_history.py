import time
import os
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
# from models.networks import DistillLoss
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

teacher_tested = False

def dir_psnr(A, B):
    psnr = PeakSignalNoiseRatio(data_range =1.)
    psnrs = []
    a_sorted = sorted(os.listdir(A))
    b_sorted = sorted(os.listdir(B))
    assert  a_sorted == b_sorted, "directory files are not identical"
    for i in tqdm(range(len(a_sorted))):
        a_img = torchvision.io.read_image(A + '/' + a_sorted[i]).cuda()/255.0
        b_img = torchvision.io.read_image(B + '/' + b_sorted[i]).cuda()/255.0
        psnr.update(a_img, b_img)
        item = psnr.compute()
        psnrs.append(item.cpu())
        psnr.reset()
    return np.array(psnrs).mean()

def dir_tpsnr(A, B):
    psnr = PeakSignalNoiseRatio(data_range =1.)
    psnrs = []
    a_sorted = sorted(os.listdir(A))
    b_sorted = sorted(os.listdir(B))
    assert  a_sorted == b_sorted, "directory files are not identical"
    for i in tqdm(range(1,len(a_sorted))):

        a_img = torchvision.io.read_image(A + '/' + a_sorted[i]).cuda()/255.0
        old_a_img = torchvision.io.read_image(A + '/' + a_sorted[i-1]).cuda()/255.0
        b_img = torchvision.io.read_image(B + '/' + b_sorted[i]).cuda()/255.0
        old_b_img = torchvision.io.read_image(B + '/' + b_sorted[i-1]).cuda()/255.0

        psnr.update(a_img- old_a_img, b_img-old_b_img)

        item = psnr.compute()
        psnrs.append(item.cpu())
        psnr.reset()

    return np.array(psnrs).mean()



def dir_lpips(A, B):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    lpipz = []
    a_sorted = sorted(os.listdir(A))
    b_sorted = sorted(os.listdir(B))
    assert  a_sorted == b_sorted, "directory files are not identical"
    for i in tqdm(range(len(a_sorted))):
        a_img = torchvision.io.read_image(A + '/' + a_sorted[i]).cuda()/255.0
        b_img = torchvision.io.read_image(B + '/' + b_sorted[i]).cuda()/255.0
        
        with torch.no_grad():
            item = loss_fn_alex(a_img*2 -1, b_img*2 -1).item()
        lpipz.append(item)
    return np.array(lpipz).mean()


def dir_fid(A,B):
    fid = os.popen(f'python -m pytorch_fid {A} {B}').read()
    fid = (float(fid.replace('FID:','').strip()))
    return fid


opt = TrainOptions().parse()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idss


iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)


opt.norm = 'instance'
model = create_model(opt)
import torch.nn as nn
def has_batchnorm(model):
    for module in model.modules():
        print(module)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False

# Example usage

## Hardcoding affine = False 

import sys

# Example of removing specific arguments
args_to_remove = [

('--display_freq', str(opt.display_freq)),

('--experiment_name', opt.experiment_name),

('--niter', str(opt.niter)),

('--niter_decay', str(opt.niter_decay)),

('--save_epoch_freq', str(opt.save_epoch_freq)),

('--resume_distill_epoch', str(opt.resume_distill_epoch)),

('--alpha', str(opt.alpha)),

('--alpha1', str(opt.alpha1)),

('--alpha2', str(opt.alpha2)),

('--alpha3', str(opt.alpha3)),

('--alpha4', str(opt.alpha4)),

('--alpha5', str(opt.alpha5)),

('--aim_repo', str(opt.aim_repo)),

('--no_ganFeat_loss', str(opt.no_ganFeat_loss)),

('--accum_iter', str(opt.accum_iter)),


]

print(" args:", sys.argv)
filtered_args = []
skip_next = False

for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    # if i + 1 < len(sys.argv):
    #     print(arg,sys.argv[i + 1])
    # for key, value in args_to_remove:
    #     if key==arg:
    #         print(value, sys.argv[i+1], sys.argv[i+1]==value)
    # print([f'\n key = {key}' for key, v in args_to_remove])

    if any(arg == key and (i + 1 < len(sys.argv) and sys.argv[i + 1] == value) for key, value in args_to_remove):
        skip_next = True  # Skip the next value since it's part of the key-value pair to remove
    else:
        print(f'arg added = {arg}')
        if arg not in ['--resume_distill_epoch','--no_ganFeat_loss','--wavelet', '--teacher_adv', '--teacher_feat', '--teacher_vgg', '--aim_repo']:
            filtered_args.append(arg)

sys.argv = filtered_args


test_opt = TestOptions().parse(save=False)
test_opt.no_flip=True
test_opt.loadSize = 1024
test_opt.fineSize = 1024
test_opt.batchSize =1
test_opt.serial_batches = True
test_opt.phase = 'test'

test_opt.use_encoded_image = True



val_opt = test_opt
val_opt.phase = 'val'
val_data_loader = CreateDataLoader(val_opt)
val_dataset = val_data_loader.load_data()
val_dataset_size = len(val_data_loader)


visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D


# total_steps = (start_epoch-1) * dataset_size + epoch_iter

# display_delta = total_steps % opt.display_freq
# print_delta = total_steps % opt.print_freq
# save_delta = total_steps % opt.save_latest_freq

if opt.resume_distill_epoch != 0:
    opt.resume_repo = opt.name

    my_string = "Hello, PyTorch!"

    # Save the string in a dictionary format
    string = torch.load(f'checkpoints/{opt.resume_repo}/aim_strings.pth')
    aim_id = string['aim_id']
    repo = string['repo']

    run = Run(
        # run_hash=aim_id,
        repo=repo,
        experiment=opt.experiment_name,
        log_system_params =True
    )


    g_checkpoint = torch.load(f'checkpoints/{opt.resume_repo}/epoch_{opt.resume_distill_epoch}_netG.pth', map_location = 'cuda:0')
    d_checkpoint = torch.load(f'checkpoints/{opt.resume_repo}/epoch_{opt.resume_distill_epoch}netD.pth', map_location = 'cuda:0')

    og_checkpoint = torch.load( f'checkpoints/{opt.resume_repo}/epoch_{opt.resume_distill_epoch}_optim-0.pth', map_location = 'cuda:0')
    od_checkpoint = torch.load(f'checkpoints/{opt.resume_repo}/epoch_{opt.resume_distill_epoch}_optim-1.pth', map_location = 'cuda:0')

    model.module.netG.load_state_dict(g_checkpoint)
    model.module.netD.load_state_dict(d_checkpoint)


    
    model.module.optimizer_G.load_state_dict(og_checkpoint)
    model.module.optimizer_D.load_state_dict(od_checkpoint)
    new_start_epoch = opt.resume_distill_epoch



else:
    new_start_epoch = start_epoch
    run = Run(
        repo=opt.aim_repo,
        experiment=opt.experiment_name,
        log_system_params =True
    )



# strings = {'aim_id': run.hash, 'repo': opt.aim_repo}
# torch.save(strings, f'checkpoints/{opt.name}/aim_strings.pth')


# loading the teacher... 
for epoch in range(new_start_epoch, opt.niter + opt.niter_decay + 1):
    g_checkpoint = torch.load(f'checkpoints/{opt.name}/epoch_{epoch}_netG.pth', map_location = 'cuda:0')
    model.module.netG.load_state_dict(g_checkpoint)


    # on the validation dataset
    os.makedirs('fake', exist_ok=True)
    os.makedirs('real', exist_ok=True)
    os.makedirs('input', exist_ok=True)

    os.makedirs(f'fake/{opt.experiment_name}_validation', exist_ok=True)
    os.makedirs(f'real/{opt.experiment_name}_validation', exist_ok=True)
    os.makedirs(f'input/{opt.experiment_name}_validation', exist_ok=True)

    model = model.eval()
    for i, data in tqdm(enumerate(val_dataset)):   
        ############## Forward Pass ######################
        with torch.no_grad():
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['image']), Variable(data['feat']),infer=True)
            gen1 = (util.tensor2im(generated.data[0]))
            real1 = (util.tensor2im(data['image'][0]))

            cv2.imwrite(f'fake/{opt.experiment_name}_validation/{i}.png', gen1[:,:,::-1])
            cv2.imwrite(f'real/{opt.experiment_name}_validation/{i}.png', real1[:,:,::-1])



    del gen1
    del real1


    
    torch.cuda.empty_cache()
    fid = dir_fid(f'fake/{opt.experiment_name}_validation', f'real/{opt.experiment_name}_validation')
    lpipzz = dir_lpips(f'fake/{opt.experiment_name}_validation', f'real/{opt.experiment_name}_validation')
    psnr = dir_psnr(f'fake/{opt.experiment_name}_validation', f'real/{opt.experiment_name}_validation')
    tpsnr = dir_tpsnr(f'fake/{opt.experiment_name}_validation', f'real/{opt.experiment_name}_validation')
    
    run.track(psnr, name='validation PSNR', step = epoch)
    run.track(tpsnr, name='validation tPSNR', step = epoch)
    run.track(fid, name = 'validation FID', step = epoch)
    run.track(lpipzz, name = 'validation LPIPS', step = epoch)
