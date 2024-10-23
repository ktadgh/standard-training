import time
import os
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
# from models.networks import DistillLoss
from thop import profile
from models.networks import DistillLoss, OFLoss
from torch.profiler import profile, record_function, ProfilerActivity

import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
from aim import Run, Image
# torch.set_float32_matmul_precision('high')
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
# temp_loss = OFLoss()

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
    fid = os.popen(f'python -m pytorch_fid --device cuda:0 {A} {B} ').read()
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


import sys

test_opt = opt
test_opt.no_flip=True
test_opt.loadSize = 1024
test_opt.fineSize = 1024
test_opt.batchSize =1
test_opt.serial_batches = True
test_opt.phase = 'test'

test_opt.use_encoded_image = True
test_data_loader = CreateDataLoader(test_opt)

test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_dataset)

# setting validation options
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


total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

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



strings = {'aim_id': run.hash, 'repo': opt.aim_repo}
torch.save(strings, f'checkpoints/{opt.name}/aim_strings.pth')


# loading the teacher... 
for epoch in range(new_start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    losses_G= 0
    losses_D= 0

    j = -1
    for i, data in enumerate(tqdm(dataset), start=epoch_iter):
        j +=1
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        if opt.debug_script:
            if j > 10:
                break
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        if epoch ==11:
            break
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        

        ############## Forward Pass ######################
        losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']),Variable(data['image']), Variable(data['feat']),infer=True, teacher_adv = opt.teacher_adv,
            teacher_feat = opt.teacher_feat,teacher_vgg = opt.teacher_vgg)



        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))


        
        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
        distloss = 0



        # with record_function("OFLoss"):
        #     if opt.alpha_temporal != 0:

        #         gt1 = data['image'].cuda()
        #         gt2 = data['next_image'].cuda()

        #         with torch.no_grad(): generated2 = model.module.netG(data['new_label'].cuda())
        #         if i == 0:
        #             tl = temp_loss(generated,generated2,gt1,gt2)
        #         else:
        #             tl = temp_loss(generated,generated2,gt1,gt2)

        #         loss_G += tl*opt.alpha_temporal


        model.module.netD.requires_grad_(False)
        loss_G.backward()
        model.module.netD.requires_grad_(True)

        if j % opt.accum_iter == 0:
            optimizer_G.step()
            optimizer_G.zero_grad()

        model.module.netG.requires_grad_(False)
        loss_D.backward()
        model.module.netG.requires_grad_(True)

        if j % opt.accum_iter == 0:
            optimizer_D.step()
            optimizer_D.zero_grad()


            ############### Backward Pass ####################
            # update generator weights



        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 


        if j %  100 ==0:
            # tracking metrics with AIM
            run.track(loss_D.detach(), name = 'Disriminator loss')
            run.track(loss_dict['G_GAN'].detach(), name = 'GAN loss (default is hinge)')
            run.track(loss_dict.get('G_GAN_Feat',0).detach(), name = 'Feature Loss')
            run.track(loss_dict.get('G_VGG',0).detach(), name = 'VGG loss')
            
        # prof.export_chrome_trace("trace.json")
        # raise ValueError('Exported')
        
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))



    if epoch % opt.save_epoch_freq == 0:
        torch.save(model.module.optimizer_G.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_optim-0.pth')
        torch.save(model.module.optimizer_D.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_optim-1.pth')
        torch.save(model.module.netG.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_netG.pth')
        torch.save(model.module.netD.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}netD.pth')

        os.makedirs('fake', exist_ok=True)
        os.makedirs('real', exist_ok=True)
        os.makedirs(f'fake/{opt.experiment_name}', exist_ok=True)
        os.makedirs(f'real/{opt.experiment_name}', exist_ok=True)

        for i, data in tqdm(enumerate(test_dataset)):   
            if opt.debug_script:
                if i > 55:
                    break
            elif i > 2000:
                break
            if i <= 1:
                labels = data['label']  # A list of tensors
                images = data['image']  # A list of tensors

                tensors = labels
                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())

            else:
                labels = data['label'].cuda()  # A list of tensors
                images = data['image']  # A list of tensors

                tensors = labels
                gen1 = (util.tensor2im(generated[0]))
                cv2.imwrite(f'pregen/{opt.experiment_name}/{i}.png', gen1[:,:,::-1])

                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())
                gen1 = (util.tensor2im(generated[0]))
                real1 = (util.tensor2im(data['image'][0].squeeze()))
                cv2.imwrite(f'fake/{opt.experiment_name}/{i}.png', gen1[:,:,::-1])
                cv2.imwrite(f'real/{opt.experiment_name}/{i}.png', real1[:,:,::-1])



            
        torch.cuda.empty_cache()

        fid = dir_fid(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        lpipzz = dir_lpips(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        psnr = dir_psnr(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        tpsnr = dir_tpsnr(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        
        run.track(psnr, name='PSNR')
        run.track(tpsnr, name='tPSNR')
        run.track(fid, name = 'FID')
        run.track(lpipzz, name = 'LPIPS')




        os.makedirs('fake', exist_ok=True)
        os.makedirs('real', exist_ok=True)
        os.makedirs(f'fake/{opt.experiment_name}_val', exist_ok=True)
        os.makedirs(f'real/{opt.experiment_name}_val', exist_ok=True)


        for i, data in tqdm(enumerate(val_dataset)):   
            if opt.debug_script:
                if i > 55:
                    break
            elif i > 2000:
                break

            if i <= 1:
                labels = data['label'].cuda()  # A list of tensors

                tensors = labels
                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())

            else:
                labels = data['label'].cuda()  # A list of tensors

                tensors = labels
                gen1 = (util.tensor2im(generated[0]))
                cv2.imwrite(f'pregen/{opt.experiment_name}/{i}.png', gen1[:,:,::-1])

                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())
                gen1 = (util.tensor2im(generated[0]))
                real1 = (util.tensor2im(data['image'][0].squeeze()))
                cv2.imwrite(f'fake/{opt.experiment_name}_val/{i}.png', gen1[:,:,::-1])
                cv2.imwrite(f'real/{opt.experiment_name}_val/{i}.png', real1[:,:,::-1])





        del gen1
        del real1

        fid = dir_fid(f'fake/{opt.experiment_name}_val', f'real/{opt.experiment_name}_val')
        lpipzz = dir_lpips(f'fake/{opt.experiment_name}_val', f'real/{opt.experiment_name}_val')
        psnr = dir_psnr(f'fake/{opt.experiment_name}_val', f'real/{opt.experiment_name}_val')
        tpsnr = dir_tpsnr(f'fake/{opt.experiment_name}_val', f'real/{opt.experiment_name}_val')
        
        run.track(psnr, name='validation PSNR')
        run.track(tpsnr, name='validation tPSNR')
        run.track(fid, name = 'validation FID')
        run.track(lpipzz, name = 'validation LPIPS')


    model = model.train()
    torch.cuda.empty_cache()
    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()