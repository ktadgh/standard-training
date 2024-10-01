import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import k_diffusion as K
from resample2d import Resample2d

import sys
sys.path.append('/home/ubuntu/tdist-flat')
from flownet.models1 import FlowNet2

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance',config_path='NONE', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, config_path=config_path)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

# from thop import profile
def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    # x = torch.randn(1, 10, 1024, 1024).cuda()
    # mynetD = MultiscaleDiscriminator(10, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat).cuda()
    # macs10, params = profile(mynetD, inputs=(x,))

    # x = torch.randn(1, 17, 1024, 1024).cuda()
    # mynetD = MultiscaleDiscriminator(17, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat).cuda()
    # macs17, params = profile(mynetD, inputs=(x,))

    # raise ValueError(macs10, macs17)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect',config_path='NONE', teacher = False):
        super().__init__()
        if teacher == False:
            config = K.config.load_config(config_path)
        else:
            config = K.config.load_config('/home/tadgh720x/Documents/distillation/transformer-distillation/configs/small-swin-students/config_oxford_flowers_shifted_window.json')
        self.model = K.config.make_model(config).cuda()

        self.final_activation_function = nn.Tanh()
        self.projector1 = nn.Linear(768, 768, bias=False)
        self.projector2 = nn.Linear(1024, 1024, bias=False)
        self.projector3 = nn.Linear(1024, 1024, bias=False)
        self.projector4 = nn.Linear(384, 384, bias=False)
        self.projector5 = nn.Linear(128, 128, bias=False)

        self.alpha1 = nn.Parameter(torch.tensor([1.]))
        self.alpha2 = nn.Parameter(torch.tensor([1.]))
        self.alpha3 = nn.Parameter(torch.tensor([1.]))
        self.alpha4 = nn.Parameter(torch.tensor([1.]))
        self.alpha5 = nn.Parameter(torch.tensor([1.]))
        self.alpha_sum = nn.Parameter(torch.tensor([5.]))

    def forward(self, input):
        cst = torch.ones((input.shape[0]), device=input.device)
        return self.final_activation_function(self.model(input, cst))           
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


from kornia.enhance import ZCAWhitening,zca_whiten


from aim import Image
import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked),
            eye, 
            upper=False
            )
        
        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)


class DistillLoss(torch.nn.Module):
    def __init__(self, teacher, student,batch_size=1, layer = 5):
        super().__init__()
        self.teacher = teacher.cuda()
        self.student = student.cuda()
        self.whitener = Whitening2d(batch_size, eps = 1).cuda()
        self.layer = layer

    def forward(self,x,run, whitening=True):
        sxs = []
        txs = []
        # _ = self.student.netG(x.cuda())

        # with torch.no_grad():
        #     _ = self.teacher.netG(x.cuda())

        projectors = [self.student.netG.projector1,self.student.netG.projector2,self.student.netG.projector3,self.student.netG.projector4,self.student.netG.projector5]
        student_patches = [self.student.netG.model.patches_for_distillation1,self.student.netG.model.patches_for_distillation2,
                           self.student.netG.model.patches_for_distillation3, self.student.netG.model.patches_for_distillation4,
                           self.student.netG.model.patches_for_distillation5]
        
        teacher_patches = [self.teacher.netG.model.patches_for_distillation1,self.teacher.netG.model.patches_for_distillation2,
                           self.teacher.netG.model.patches_for_distillation3, self.teacher.netG.model.patches_for_distillation4,
                           self.teacher.netG.model.patches_for_distillation5]

        layer_index = self.layer-1
        # print(layer_index, flush=True)
        sxs = projectors[layer_index](student_patches[layer_index].mean(dim=(1,2)))
        txs = teacher_patches[layer_index].mean(dim=(1,2))
        
        sqrt_n = torch.sqrt(torch.tensor(txs.shape[0]-1, dtype=torch.float64))
        if whitening==True:
            wt = self.whitener(txs.T).T/sqrt_n
        else:
            wt = txs

        loss = torch.norm(torch.abs(sxs-wt), p='fro')**2
        return loss


class DistillLossNoPool(torch.nn.Module):
    def __init__(self, teacher, student, batch_size =1):
        super().__init__()
        self.teacher = teacher.cuda()
        self.student = student.cuda()
        self.whitener = Whitening2d(batch_size, eps = 1).cuda()

    def forward(self,x,run, whitening=True):
        sxs = []
        txs = []


        _ = self.student.netG(x.cuda())
        sxs = self.student.netG.projector(self.student.netG.model.patches_for_distillation)

        with torch.no_grad():
            _ = self.teacher.netG(x.cuda())
            txs = self.teacher.netG.model.patches_for_distillation.detach()
            sqrt_n = torch.sqrt(torch.tensor(txs.shape[0]-1, dtype=torch.float64))
            if whitening==True:
                wt = self.whitener(txs.T).T/sqrt_n
            else:
                wt = txs

        loss = torch.norm(torch.abs(sxs-wt), p='fro')**2
        return loss


class OFLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.camera_matrix = torch.load('camera_matrix.pth').cuda()
        self.criterion = nn.MSELoss(size_average=True)

        class Args:
            def __init__(self, dictionary):
                for key in dictionary:
                    setattr(self, key, dictionary[key])

        args_dict = {
            "fp16": False,
            "rgb_max": 1.
        }


        # Convert the dictionary to an object
        args = Args(args_dict)

        self.flownet = FlowNet2(args).cuda()
        chkpt=torch.load('/home/ubuntu/FlowNet2_checkpoint.pth.tar')
        self.flownet.load_state_dict(chkpt['state_dict'])
        self.flownet = self.flownet.eval()
        self.flownet.requires_grad=False
        self.flownet = self.flownet.to('cuda:0')


    def forward(self,im1,im2, gt1,gt2, run = None, device = 'cuda'):
        flow_warping = Resample2d().to(device)
        input = torch.stack([gt2, gt1],dim =2)

        # with torch.no_grad():
        try:
            flow_i21 = (self.flownet(input.to('cuda:0'))) # flow from model input1 to 2
        except:
            raise ValueError(input.shape)
                
        gt1 = gt1.to('cuda:0')
        flow_i21 = flow_i21.to('cuda:0')
        warp_i1 = flow_warping(gt1, flow_i21)# flow warped gt

        if run is not None:
            warped_im = (warp_i1[0]+1)*0.5
            warped_im = (warped_im*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(warped_im)
            run.track(image, name='warped_im')

            gt1_im = (gt1[0]+1)*0.5
            gt1_im = (gt1_im*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(gt1_im)
            run.track(image, name='Ground Truth 1')

            gt2_im = (gt2[0]+1)*0.5
            gt2_im = (gt2_im*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(gt2_im)
            run.track(image, name='Ground Truth 2')

            im1_im = (im1[0]+1)*0.5
            im1_im = (im1_im*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(im1_im)
            run.track(image, name='Input 1')

            im2_im = (im2[0]+1)*0.5
            im2_im = (im2_im*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(im2_im)
            run.track(image, name='Input 2')

        diff = (gt2.to('cuda:0') - warp_i1.to('cuda:0'))
        sumdiff = torch.sum(diff, dim=1)
        summdiff2 = sumdiff.pow(2)
        mask = torch.exp(-50.0 * summdiff2).unsqueeze(1).to('cuda:0')
        
        if run is not None:
            mask_im = mask[0].squeeze()
            mask_im = (mask_im*255).detach().cpu().numpy().astype(np.uint8)
            mask_im = Image(mask_im)
            run.track(mask_im, name='mask')

        with torch.no_grad():          
            warp_o1 = flow_warping(im1.to('cuda:0'), flow_i21.to('cuda:0')).detach() # flow warped model output
        if run is not None:
            warped_out = warp_i1 * mask
            warped_out = (warped_out[0])
            warped_out_im =  (warped_out*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(warped_out_im)
            run.track(image, name='masked warped gt1')

            equal_tensor = (warp_i1.cuda() == gt2.cuda())
            equal_tensor = (equal_tensor[0]+1)*0.5
            equal_im = (equal_tensor*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(equal_im)
            run.track(image, name='equal_tensor')

            masked_gt2 = gt2.cuda() * mask.cuda()
            masked_gt2 = (masked_gt2[0])
            masked_gt2_im = (masked_gt2*255).detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
            image = Image(masked_gt2_im)
            run.track(image, name='masked_gt2')

        return  self.criterion(im2 * mask, warp_o1.to(device) * mask).to(device)