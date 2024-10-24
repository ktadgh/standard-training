import time
import os
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.networks import DistillLoss, OFLoss
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
import wandb
import random

random.seed(1)
torch.manual_seed(1)

temp_loss = OFLoss()
mse = torch.nn.MSELoss(size_average=True).cuda() 


torch.autograd.set_detect_anomaly(True)

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
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
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

('--alpha_temporal', str(opt.alpha_temporal)),

('--delta_loss', str(opt.delta_loss)),

('--accum_iter', str(opt.accum_iter)),

('--cutoff', str(opt.cutoff)),

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
        if arg not in ['--resume_distill_epoch', '--teacher_adv', '--teacher_feat', '--teacher_vgg', '--aim_repo', '--delta_loss']:
            filtered_args.append(arg)

sys.argv = filtered_args[:1] + filtered_args[3:]


test_opt = TestOptions().parse(save=False)
test_opt.no_flip=True
test_opt.loadSize = 1024
test_opt.fineSize = 1024
test_opt.batchSize =1
test_opt.serial_batches = True
test_opt.phase = 'val'

test_opt.use_encoded_image = True
test_data_loader = CreateDataLoader(test_opt)

test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_dataset)
from torch.profiler import profile, record_function, ProfilerActivity

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
        repo='nothing',
        experiment=opt.experiment_name,
        log_system_params =True
    )


    # To load the string back

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


wandb.init(
    # Set the project where this run will be logged

    project="1024-lit8-18-09",

    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)

    name=opt.experiment_name,

    # Track hyperparameters and run metadata

    config={
        "learning_rate": 0.02,
    },

    resume="allow")

strings = {'aim_id': run.hash, 'repo': opt.aim_repo}
torch.save(strings, f'checkpoints/{opt.name}/aim_strings.pth')


for epoch in range(new_start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    losses_G= 0
    losses_D= 0
    old_data = 0
    old_model_data = 0

    j = -1
    # with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:  
    for i, data in enumerate(tqdm(dataset),start=epoch_iter):
        j +=1
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        labels = ((data['label']))  # A list of tensors
        images = (data['image'])  # A list of tensors

        # Use slicing and torch.cat to concatenate pairs
        tensors = torch.cat([torch.cat([labels[:-1],labels[1:]], dim=1)], dim=0)

        # Extract the previous elements for true_tensors
        true_tensors = (images[1:])

        data['image'] = true_tensors
        data['label'] = tensors

        # with record_function("forward_pass"):
            ############## Forward Pass ######################
        if j == 0:
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']),Variable(data['image']), Variable(data['feat']),infer=True, teacher_adv = opt.teacher_adv,
                teacher_feat = opt.teacher_feat,teacher_vgg = opt.teacher_vgg)
        else:
            losses, generated = model.forward(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['image']), Variable(data['feat']),infer=True, teacher_adv = opt.teacher_adv,
                teacher_feat = opt.teacher_feat,teacher_vgg = opt.teacher_vgg)       

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))


        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) +loss_dict['G_GAN']
        distloss = 0
        # _ = model.module.netG(data['label'].cuda())

        # distloss = opt.alpha1*dloss1(data['label'], run, whitening= False) + opt.alpha2*dloss2(data['label'], run, whitening= False) + opt.alpha3*dloss3(data['label'], run, whitening= False)
        # + opt.alpha4*dloss4(data['label'], run, whitening= False) + opt.alpha5*dloss5(data['label'], run, whitening= False)
        # # sum = (model.module.netG.alpha1+model.module.netG.alpha2+model.module.netG.alpha3+model.module.netG.alpha4+model.module.netG.alpha5)
        # loss_G += ( distloss.squeeze() )* opt.alpha


        ###### Temporal Loss #######
        gt2 = images[2:]
        gt1 = images[1:-1]

        model_data = generated

        im1 = model_data[:-1].detach()
        im2 = model_data[1:]

        if opt.alpha_temporal != 0:
            if opt.delta_loss:
                delta_im = im2 - im1
                delta_gt = gt2 - gt1
                tl = mse(delta_im.cuda(), delta_gt.cuda())
            else:
                if j == 0:
                    tl = temp_loss(im1,im2,gt1,gt2, run)
                else:
                    tl = temp_loss(im1,im2,gt1,gt2, None)

            if j % 50 ==0:
                run.track(tl*opt.alpha_temporal, name = 'Temporal Loss', step = epoch)

            loss_G =loss_G+ tl*opt.alpha_temporal



        ############### Backward Pass ####################
        # d_grads_before = [param.grad.clone() if param.grad is not None else None for param in model.module.netD.parameters() ]
        model.module.netD.requires_grad_(False)
        loss_G.backward()
        model.module.netD.requires_grad_(True)
        # d_grads_after = [param.grad.clone() if param.grad is not None else None for param in model.module.netD.parameters() ]
        # assert all(torch.allclose(before, after) for before, after in zip(d_grads_before, d_grads_after) if (before is not None) or (after is not None)), "Discriminator gradients altered"

        if j % opt.accum_iter == 0:
            optimizer_G.step()
            optimizer_G.zero_grad()

        # g_grads_before = [param.grad.clone() if param.grad is not None else None for param in model.module.netG.parameters()]
        model.module.netG.requires_grad_(False)
        loss_D.backward()
        model.module.netG.requires_grad_(True)
        # g_grads_after = [param.grad.clone()  if param.grad is not None else None for param in model.module.netG.parameters()]
        # assert all(torch.allclose(before, after) for before, after in zip(g_grads_before, g_grads_after)if (before is not None) or (after is not None)), "Generator gradients altered"

        if j % opt.accum_iter == 0:
            optimizer_D.step()
            optimizer_D.zero_grad()  


        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 


        if j ==0:
            gen1 = Image(util.tensor2im(generated.data[0]))
            real1 = Image(util.tensor2im(data['image'][0]))
            inp1 = Image(util.tensor2im(data['label'][0]))
            # t1 = Image(util.tensor2im(teacher_image[0]))

            run.track(gen1, name = 'generated image')
            run.track(real1, name = 'real image')
            run.track(inp1, name = 'input image')

            gen1 = wandb.Image(util.tensor2im(generated.data[0]))
            real1 = wandb.Image(util.tensor2im(data['image'][0]))
            inp1 = wandb.Image(util.tensor2im(data['label'][0][:3]))
            # raise ValueError(data['label'].shape)
            depth1 = wandb.Image((data['label'][0,3]))
            
            normal1 = wandb.Image((data['label'][0,4:7]*0.5 + 0.5))

            inp2 = wandb.Image(util.tensor2im(data['label'][0,7:10]))
            depth2 = wandb.Image((data['label'][0,10]))
            normal2 = wandb.Image(util.tensor2im(data['label'][0,11:14]*0.5 + 0.5))

            try:
                wandb.log({"generated image": gen1},step=epoch)
                wandb.log({"real image": real1},step=epoch)
                wandb.log({"input image": inp1},step=epoch)
                wandb.log({"depth": depth1},step=epoch)
                wandb.log({"normal": normal1},step=epoch)

                wandb.log({"input image 2": inp2},step=epoch)
                wandb.log({"depth 2": depth2},step=epoch)
                wandb.log({"normal 2": normal2},step=epoch)
            
            except:
                pass

            gen1 = Image(util.tensor2im(generated.data[0]))
            real1 = Image(util.tensor2im(data['image'][0]))
            inp1 = Image(util.tensor2im(data['label'][0][:3]))
            # raise ValueError(data['label'].shape)
            depth1 = Image((data['label'][0,3]))
            
            normal1 = Image((data['label'][0,4:7]*0.5 + 0.5))

            inp2 = Image(util.tensor2im(data['label'][0,7:10]))
            depth2 = Image((data['label'][0,10]))
            normal2 = Image(util.tensor2im(data['label'][0,11:14]*0.5 + 0.5))

            run.track(depth1, name = "Depth 1",step=epoch)
            run.track(normal1, name = "Normal 1",step=epoch)

            run.track(inp2, name = "Input 2",step=epoch)
            run.track(depth2, name = "Depth 2",step=epoch)
            run.track(normal2, name = "Normal 2",step=epoch)

            # tracking metrics with AIM
        if j % 50 == 0:
            run.track(loss_D.detach(), name = 'Disriminator loss')
            run.track(loss_dict['G_GAN'].detach(), name = 'GAN loss (default is hinge)')
            run.track(loss_dict.get('G_GAN_Feat',0).detach(), name = 'Feature Loss')
            run.track(loss_dict.get('G_VGG',0).detach(), name = 'VGG loss')

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))



    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        torch.save(model.module.optimizer_G.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_optim-0.pth')
        torch.save(model.module.optimizer_D.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_optim-1.pth')
        torch.save(model.module.netG.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}_netG.pth')
        torch.save(model.module.netD.state_dict(), f'checkpoints/{opt.name}/epoch_{epoch}netD.pth')

        os.makedirs('fake', exist_ok=True)
        os.makedirs('real', exist_ok=True)
        os.makedirs('pregen', exist_ok=True)
        os.makedirs(f'fake/{opt.experiment_name}', exist_ok=True)
        os.makedirs(f'real/{opt.experiment_name}', exist_ok=True)
        os.makedirs(f'pregen/{opt.experiment_name}', exist_ok=True)

        model = model.eval()
        generated = torch.zeros(1,3,1024,1024).cuda()

        previous_label = torch.zeros(1,14,1024,1024).cuda()
        for i, data in tqdm(enumerate(test_dataset)):  
            if i == 0:
                previous_label = (data['label'].cuda())
            if i <= 1:
                labels = (data['label']).cuda()  # A list of tensors
                images = (data['image']).cuda()  # A list of tensors

                tensors = torch.cat([torch.cat([previous_label,labels], dim=1)], dim=0)

                previous_label = labels
                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())

            else:
                labels = (data['label'].cuda())  # A list of tensors
                images = (data['image'].cuda())  # A list of tensors

                tensors = torch.cat([torch.cat([previous_label, labels], dim=1)], dim=0)
                previous_label = labels
                gen1 = (util.tensor2im(generated[0]))
                cv2.imwrite(f'pregen/{opt.experiment_name}/{i}.png', gen1[:,:,::-1])

                with torch.no_grad():
                    generated = model.module.netG(tensors.cuda())
                gen1 = (util.tensor2im(generated[0]))
                real1 = (util.tensor2im(data['image'][0].squeeze()))
                cv2.imwrite(f'fake/{opt.experiment_name}/{i}.png', gen1[:,:,::-1])
                cv2.imwrite(f'real/{opt.experiment_name}/{i}.png', real1[:,:,::-1])

        del gen1
        del real1


        
        torch.cuda.empty_cache()
        fid = dir_fid(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        lpipzz = dir_lpips(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        psnr = dir_psnr(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        tpsnr = dir_tpsnr(f'fake/{opt.experiment_name}', f'real/{opt.experiment_name}')
        
        run.track(psnr, name='PSNR')
        run.track(tpsnr, name='tPSNR')
        run.track(fid, name = 'FID')
        run.track(lpipzz, name = 'LPIPS')

        try:
            wandb.log({"PSNR": psnr, "tPSNR": tpsnr, "FID":fid, "LPIPS": lpipzz},step=epoch)
        except:
            pass


    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()