import torch
import sys
import k_diffusion as K
import torch._C._onnx as _C_onnx
from torchinfo import summary
import cv2
import numpy as np
import torchvision

import pynvml
import onnxruntime as ort
import os
import torch.nn as nn
from PIL import Image

for file in os.listdir('/home/ubuntu/transformer-distillation/configs/hdit-shifted-windows'):
    if 'patchsize' not in file:
        continue
    file = 'original-swin'
    config = '/home/ubuntu/transformer-distillation/configs/hdit_shifted_window.json'#'/home/ubuntu/transformer-distillation/configs/hdit-shifted-windows/'+ file
    config = K.config.load_config(config)

    model = K.config.make_model(config).cuda()

    # dct = torch.load("../200_net_G_hdit.pth")#.keys()
    # dct = {
    #     key.replace('model.', ''): value for key, value in dct.items()
    # }

    # model.load_state_dict(dct)
    final_activation_function = nn.Tanh()
    print(sum([p.numel() for p in model.parameters()]) / 1e6)
    model.eval()

    img = torch.from_numpy(np.array(Image.open("../input_0_24_62.png")))
    img = img / 255.
    img = img * 2. - 1.
    img = img.transpose(-1, -2).transpose(-2, -3).unsqueeze(0).cuda()

    cst = torch.ones((img.shape[0])).cuda() * 500
    with torch.no_grad(): out = final_activation_function(model(img, cst)[0])

    out = out / 2. + .5
    out = out.squeeze(0).transpose(0, 1).transpose(1, 2)
    out = Image.fromarray((out.data.cpu().numpy() * 255.).astype(np.uint8))
    out.save('out_shifted.png')

    class MyModel(torch.nn.Module):
        def __init__(self, model, device):
            super(MyModel, self).__init__()
            # Example: simple linear layer
            self.cst = torch.ones((1)).to(device) *500
            self.model = model

        def forward(self, img):
            x  = self.model(img,self.cst)
            return x
        
    model.eval()
    pmodel = MyModel(model, 'cpu')
    of = f'{ file.replace(".json", "") }.onnx'
    input = torch.tensor(cv2.imread('../input_0_24_62.png')).permute(2,0,1).unsqueeze(0).cuda()/255.
    img = img * 2. - 1.
    img_bs1 = torch.randn((1,3,1024,1024))
    torch.onnx.export(pmodel.cpu(), (img_bs1.cpu()), of, opset_version = 17,do_constant_folding=False, export_params=True,input_names = ['input'],output_names = ['output'], verbose=False)
    
    of = f'{ file.replace(".json", "") }_bs2.onnx'
    input = torch.tensor(cv2.imread('../input_0_24_62.png')).permute(2,0,1).unsqueeze(0).cuda()/255.
    img = img * 2. - 1.
    img_bs2 = torch.randn((2,3,1024,1024))
    torch.onnx.export(pmodel.cpu(), (img_bs2.cpu()), of, opset_version = 17,do_constant_folding=False, export_params=True,input_names = ['input'],output_names = ['output'], verbose=False)
    raise ValueError()