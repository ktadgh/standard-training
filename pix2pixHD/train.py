import time
import os
import numpy as np
from options1.train_options import TrainOptions
from options1.test_options import TestOptions

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

from data2.data_loader import CreateDataLoader
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


def dir_psnr(A, B):
    psnr = PeakSignalNoiseRatio(data_range =1.)
    psnrs = []
    a_sorted = sorted(os.listdir(A))
    b_sorted = sorted(os.listdir(B))
    assert  a_sorted == b_sorted, "directory files are not identical"
    for i in tqdm(range(len(a_sorted))):
        a_img = torchvision.io.read_image(A + '/' + a_sorted[i])/255.0
        b_img = torchvision.io.read_image(B + '/' + b_sorted[i])/255.0
        psnr.update(a_img, b_img)
        item = psnr.compute()
        psnrs.append(item)
        psnr.reset()
    return np.array(psnrs).mean()


def dir_lpips(A, B):
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    lpipz = []
    a_sorted = sorted(os.listdir(A))
    b_sorted = sorted(os.listdir(B))
    assert  a_sorted == b_sorted, "directory files are not identical"
    for i in tqdm(range(len(a_sorted))):
        a_img = torchvision.io.read_image(A + '/' + a_sorted[i])/255.0
        b_img = torchvision.io.read_image(B + '/' + b_sorted[i])/255.0
        
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

run = Run(
    repo='runs',
    experiment=opt.experiment_name,
    log_system_params =True
)

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

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)


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
for module in model.netG.modules():
    if isinstance(module, (torch.nn.InstanceNorm2d)):
        module.affine = False

for module in model.netD.modules():
    if isinstance(module, (torch.nn.InstanceNorm2d)):
        module.affine = False

import sys

# Example of removing specific arguments
args_to_remove = [
    ('--display_freq', str(opt.display_freq)),
    ('--experiment_name', opt.experiment_name),
    ('--niter', str(opt.niter)),
    ('--niter_decay', str(opt.niter_decay)),
    ('--resume_distill_epoch', str(opt.niter_decay)),
    ('--save_epoch_freq', str(opt.save_epoch_freq)),
    ('--resume_distill_epoch', str(opt.resume_distill_epoch))
]

print(" args:", sys.argv)
filtered_args = []
skip_next = False

for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    print(arg)
    print([f'\n key = {key}' for key, v in args_to_remove])

    if any(arg == key and (i + 1 < len(sys.argv) and sys.argv[i + 1] == value) for key, value in args_to_remove):
        skip_next = True  # Skip the next value since it's part of the key-value pair to remove
    else:
        filtered_args.append(arg)
sys.argv = filtered_args


test_opt = TestOptions().parse(save=False)
test_opt.no_flip=True
test_opt.resize_or_crop = ''
test_opt.batchSize =1
test_opt.serial_batches = True
test_opt.use_encoded_image = True
test_data_loader = CreateDataLoader(test_opt)

test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_dataset)

visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

if opt.resume_distill_epoch != 0:
    opt.resume_repo = opt.save_dir
    g_checkpoint = torch.load(f'{opt.resume_repo}/epoch_{opt.resume_distill_epoch}_net_G.pth', map_location = model.device)
    d_checkpoint = torch.load(f'{opt.resume_repo}/epoch_{opt.resume_distill_epoch}_net_D.pth', map_location = model.device)

    og_checkpoint = torch.load( f'{opt.save_dir}/epoch_{opt.resume_distill_epoch}_optim-0.pth', map_location = model.device)
    od_checkpoint = torch.load(f'{opt.save_dir}/epoch_{opt.resume_distill_epoch}_optim-1.pth', map_location = model.device)

    model.netG.load_state_dict(g_checkpoint)
    model.netD.load_state_dict(d_checkpoint)


    
    model.optimizer_G.load_state_dict(og_checkpoint)
    model.optimizer_D.load_state_dict(od_checkpoint)
    new_start_epoch = opt.resume_distill_epoch
else:
    new_start_epoch = start_epoch


for epoch in range(new_start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        
        ############## Forward Pass ######################
        if i == 0:
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), run,infer=True)
        else:
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), run,infer=False)       

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)


        # tracking metrics with AIM
        run.track(loss_D, name = 'Disriminator loss')
        run.track(loss_dict['G_GAN'], name = 'GAN loss (default is hinge)')
        run.track(loss_dict.get('G_GAN_Feat',0), name = 'Feature Loss')
        run.track(loss_dict.get('G_VGG',0), name = 'VGG loss')

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        optimizer_D.step()        

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 




        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
        if i ==0:
            gen1 = Image(util.tensor2im(generated.data[0]))
            real1 = Image(util.tensor2im(data['image'][0]))
            inp1 = Image(util.tensor2im(data['label'][0]))
            


            run.track(gen1, name = 'generated image')
            run.track(real1, name = 'real image')
            run.track(inp1, name = 'input image')

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))



    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.save('latest')
        model.save_networks(f'epoch_{epoch}')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
        torch.save(model.optimizer_G.state_dict(), f'{opt.save_dir}/epoch_{epoch}_optim-0.pth')
        torch.save(model.optimizer_D.state_dict(), f'{opt.save_dir}/epoch_{epoch}_optim-1.pth')

        os.makedirs('fake', exist_ok=True)
        os.makedirs('real', exist_ok=True)
        os.makedirs(f'fake/{epoch}', exist_ok=True)
        os.makedirs(f'real/{epoch}', exist_ok=True)
       
        for i, data in enumerate(test_dataset):   
            ############## Forward Pass ######################
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), run,infer=True)
            gen1 = (util.tensor2im(generated.data[0]))
            real1 = (util.tensor2im(data['image'][0]))
            cv2.imwrite(f'fake/{epoch}/{i}.png', gen1)
            cv2.imwrite(f'real/{epoch}/{i}.png', real1)
        



        fid = dir_fid(f'fake/{epoch}', f'real/{epoch}')
        lpipzz = dir_lpips(f'fake/{epoch}', f'real/{epoch}')
        psnr = dir_psnr(f'fake/{epoch}', f'real/{epoch}')
        run.track(psnr, name='PSNR')
        run.track(fid, name = 'FID')
        run.track(lpipzz, name = 'LPIPS')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()