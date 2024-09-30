import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F

max_depth = 1065353216.0


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        if opt.phase == 'train' or opt.phase == 'test':
            ### input A (label maps)
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = "/home/ubuntu/utah/1780_dataset/1780_Dataset_Input/" #os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

            ### input B (real images)
            if opt.isTrain or opt.use_encoded_image:
                dir_B = '_B' if self.opt.label_nc == 0 else '_img'
                self.dir_B = "/home/ubuntu/utah/1780_dataset/1780_Dataset_Target/"#os.path.join(opt.dataroot, opt.phase + dir_B)  
                self.B_paths = sorted(make_dataset(self.dir_B))

            ### Depth
            self.dir_depth = "/home/ubuntu/utah/1780_dataset/Depths/"#os.path.join(opt.dataroot, "PCOData")  
            self.depth_paths = sorted(make_dataset(self.dir_depth))

            ### Normal
            self.dir_normal = "/home/ubuntu/utah/1780_dataset/Normals/"#os.path.join(opt.dataroot, "Normal")  
            self.normal_paths = sorted(make_dataset(self.dir_normal))

        elif opt.phase == 'val':
            ### input A (label maps)
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = "/home/ubuntu/utah/1992024_1_6001/1780_Validation_Input/" #os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

            ### input B (real images)
            if opt.isTrain or opt.use_encoded_image:
                dir_B = '_B' if self.opt.label_nc == 0 else '_img'
                self.dir_B = "/home/ubuntu/utah/1992024_1_6001/1780_Validation_Target/"#os.path.join(opt.dataroot, opt.phase + dir_B)  
                self.B_paths = sorted(make_dataset(self.dir_B))

            ### Depth
            self.dir_depth = "/home/ubuntu/utah/1992024_1_6001/Depths/"#os.path.join(opt.dataroot, "PCOData")  
            self.depth_paths = sorted(make_dataset(self.dir_depth))

            ### Normal
            self.dir_normal = "/home/ubuntu/utah/1992024_1_6001/Normals/"#os.path.join(opt.dataroot, "Normal")  
            self.normal_paths = sorted(make_dataset(self.dir_normal))

        if opt.isTrain:
            assert len(self.A_paths) == len(self.B_paths), f"{len(self.A_paths)}_{len(self.B_paths)} \n {self.dir_A}_{self.dir_B}"
            
        assert len(self.A_paths) == len(self.depth_paths), f"{len(self.A_paths)}_{len(self.depth_paths)}"
        assert len(self.A_paths) == len(self.normal_paths), f"{len(self.A_paths)}_{len(self.normal_paths)}"

        assert [f.split('/')[-1].split('-')[0] for f in self.A_paths] == [f.split('/')[-1].split('-')[0] for f in self.depth_paths], "images and depths do not match"
        assert [f.split('/')[-1].split('-')[0] for f in self.A_paths] == [f.split('/')[-1].split('-')[0] for f in self.normal_paths], "images and normals do not match"


        if opt.isTrain:
            for i in range(len(self.A_paths)):
                assert self.A_paths[i].split('/')[-1].split('-')[0] == self.B_paths[i].split('/')[-1].split('-')[0], f"images and point cloud do not match, {self.B_paths[i].split('/')[-1].split('-')[0][0], self.A_paths[i].split('/')[-1].split('-')[0]}, i = {i}"

            # assert [f.split('/')[-1].split('-')[0] for f in self.A_paths] == [f.split('/')[-1].split('-')[0] for f in self.B_paths], f"images and point cloud do not match, {[f.split('/')[-1].split('-')[0] for f in self.A_paths][0], [f.split('/')[-1].split('-')[0] for f in self.B_paths][0]}"

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index] 
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### depth maps
        depth_path = self.depth_paths[index] 
        normal_path = self.normal_paths[index] 
        
        #depth = torch.load(depth_path) 
        depth =  np.fromfile(depth_path, dtype='int32').reshape(1024, 1024)/ max_depth
        normal = np.fromfile(normal_path, dtype='float16', count=1024*1024*4).reshape(1024, 1024, -1)

        # assert os.path.basename(A_path).replace('-color.png','') == os.path.basename(B_path).replace('-color.png','') == os.path.basename(depth_path).replace('-depth.bin','')  == os.path.basename(normal_path).replace('-normal.bin','')

        number = int(os.path.basename(A_path).replace('-color.png',''))

        if self.opt.label_nc == 0: # this is True

            depth = torch.from_numpy(depth)
            normal = torch.from_numpy(normal)


            if A_tensor.shape[-1] == 512:
                point_id = F.interpolate(point_id, size=(512, 512), mode='bicubic', align_corners=False)
                depth = F.interpolate(depth, size=(512, 512), mode='bicubic', align_corners=False)
                obj_id = F.interpolate(obj_id, size=(512, 512), mode='bicubic', align_corners=False)
                
        else:
            q=q

        p = A_tensor
        A_tensor = torch.cat((A_tensor, 
                              depth.reshape(1,1024,1024).to(A_tensor.dtype), 
                              normal[:,:,:3].permute(2,0,1).to(A_tensor.dtype)))
        
        ### if using instance maps        
        if not self.opt.no_instance: # this is False
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path, 'number': number}
        
        # returns just one element 
        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'