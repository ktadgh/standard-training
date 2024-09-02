import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F

MIN_R = -8056.0 
MAX_R = 8060.0
MIN_G = -462.5 
MAX_G = 4484.0
MIN_B = -6924.0 
MAX_B = 5504.0

def read_pos3d(filepath):

    dtype = np.dtype([
        ("R", np.float16),
        ("G", np.float16),
        ("B", np.float16),
        ("A", np.float16),
    ]
    )

    data = np.fromfile(filepath, dtype=dtype)
    
    posx = data["R"]
    posy = data["G"]
    posz = data["B"]

    return posx.reshape(1024, 1024), posy.reshape(1024, 1024), posz.reshape(1024, 1024)
    
def read_rgba(filepath):

    dtype = np.dtype([
        ("R", np.uint32),
        ("G", np.uint16),
        ("B", np.uint16),
    ]
    )

    data = np.fromfile(filepath, dtype=dtype)

    point_id = data["R"]
    depth = data["G"]
    obj_id = data["B"]

    return point_id.reshape(1024, 1024), depth.reshape(1024, 1024), obj_id.reshape(1024, 1024)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = "/home/ubuntu/utah/DYNAMIC_DATASET/Forge/images" #os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = "/home/ubuntu/utah/DYNAMIC_DATASET/UE"#os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### IDs and depth
        self.dir_depth = "/home/ubuntu/utah/DYNAMIC_DATASET/1082024_1_49999/Forge/PCOData"#os.path.join(opt.dataroot, "PCOData")  
        self.depth_paths = sorted(make_dataset(self.dir_depth))

        ### WorldPos
        self.dir_world_pos = "/home/ubuntu/utah/DYNAMIC_DATASET/1082024_1_49999/Forge/WorldPosData"#os.path.join(opt.dataroot, "WorldPosData")  
        self.worldpos_paths = sorted(make_dataset(self.dir_world_pos))

        if opt.isTrain:
            assert len(self.A_paths) == len(self.B_paths), f"{len(self.A_paths)}_{len(self.B_paths)}"
        assert len(self.A_paths) == len(self.depth_paths), f"{len(self.A_paths)}_{len(self.depth_paths)}"
        assert len(self.A_paths) == len(self.worldpos_paths), f"{len(self.A_paths)}_{len(self.worldpos_paths)}"

        
        assert [f.split('/')[-1].split('_')[0] for f in self.A_paths] == [f.split('/')[-1].split('_')[0] for f in self.depth_paths], "images and depths do not match"
        assert [f.split('/')[-1].split('_')[0] for f in self.A_paths] == [f.split('/')[-1].split('_')[0] for f in self.worldpos_paths], "images and 3dworld do not match"
        if opt.isTrain:
            assert [f.split('/')[-1].split('_')[0] for f in self.A_paths] == [f.split('/')[-1].split('_')[0] for f in self.B_paths], "images and point cloud do not match"

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
        world_path = self.worldpos_paths[index] 
        
        #depth = torch.load(depth_path) 
        point_id, depth, obj_id = read_rgba(depth_path)
        pos_x, pos_y, pos_z = read_pos3d(world_path)
        
        if self.opt.label_nc == 0:
            #transform_depth = get_transform(self.opt, params, normalize=False)
            #depth_tensor = transform_depth(depth)
            
            point_id = (point_id[None, None] / 1048575) * 2. - 1.
            depth = (depth[None, None] / 15130) * 2. - 1.
            obj_id = (obj_id[None, None] / 1986) * 2. - 1.
            point_id = torch.from_numpy(point_id)
            depth = torch.from_numpy(depth)
            obj_id = torch.from_numpy(obj_id)

            pos_x = 1.5 * (2 * (np.array(pos_x) - MIN_R) / (MAX_R - MIN_R) - 1)
            pos_y = 1.5 * (2 * (np.array(pos_y) - MIN_G) / (MAX_G - MIN_G) - 1)
            pos_z = 1.5 * (2 * (np.array(pos_z) - MIN_B) / (MAX_B - MIN_B) - 1)
            pos_x = torch.from_numpy(pos_x)[None, None]
            pos_y = torch.from_numpy(pos_y)[None, None]
            pos_z = torch.from_numpy(pos_z)[None, None]

            if A_tensor.shape[-1] == 512:
                point_id = F.interpolate(point_id, size=(512, 512), mode='bicubic', align_corners=False)
                depth = F.interpolate(depth, size=(512, 512), mode='bicubic', align_corners=False)
                obj_id = F.interpolate(obj_id, size=(512, 512), mode='bicubic', align_corners=False)
                
        else:
            q=q

        A_tensor = torch.cat((A_tensor, 
                              depth.squeeze(0).to(A_tensor.dtype), 
                              point_id.squeeze(0).to(A_tensor.dtype), 
                              obj_id.squeeze(0).to(A_tensor.dtype),
                              pos_x.squeeze(0).to(A_tensor.dtype),
                              pos_y.squeeze(0).to(A_tensor.dtype),
                              pos_z.squeeze(0).to(A_tensor.dtype)))

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'