
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table
import torchvision
import torch
from PIL import Image
import os
import numpy as np
import onnxruntime
# load the calibrated model
# state_dict = torch.load("quant_resnet50-entropy-1024.pth", map_location="cpu")
# model.load_state_dict(state_dict)





class ImageNetDataReader(CalibrationDataReader):
    def __init__(self,
                 image_folder,
                 width=1024,
                 height=1024,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 batch_size=1,
                 model_path='augmented_model.onnx',
                 input_name='data'):
        '''
        :param image_folder: image dataset folder
        :param width: image width
        :param height: image height 
        :param start_index: start index of images
        :param end_index: end index of images
        :param stride: image size of each data get 
        :param batch_size: batch size of inference
        :param model_path: model name and path
        :param input_name: model input name
        '''

        self.image_folder = image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.datasize = 0
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(self.image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1
        self.batch_size = batch_size
        self.input_name = input_name
        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sess_options.enable_mem_pattern = False
        self.sess_options.use_deterministic_compute = True
        self.sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.enable_cpu_mem_arena = False
    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(self.model_path, self.sess_options,providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width
        height = self.width
        nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(self.image_folder, height, width,
                                                                                  self.start_index, self.stride)
        input_name = self.input_name

        data = []
        for i in range(len(nchw_data_list)):
            nhwc_data = nchw_data_list[i]
            file_name = filename_list[i]
            data.append({input_name: nhwc_data})
        return data

    def load_batches(self):
        width = self.width
        height = self.height
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index
            nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(
                self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            for i in range(len(nchw_data_list)):
                nhwc_data = np.squeeze(nchw_data_list[i], 0)
                nchw_data_batch.append(nhwc_data)
            batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
            data = {input_name: batch_data}

            batches.append(data)

        return batches

    def preprocess_imagenet(self, images_folder, height, width, start_index=0, size_limit=0):
        '''
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter start_index: image index to start with   
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        '''
        def preprocess_images(input, channels=3, height=1024, width=1024):
            image = input.resize((width, height), Image.Resampling.LANCZOS)
            input_data = np.asarray(image).astype(np.float32)
            if len(input_data.shape) != 2:
                input_data = input_data.transpose([2, 0, 1])
            else:
                input_data = np.stack([input_data] * 3)
            mean = np.array([0.079, 0.05, 0]) + 0.406
            std = np.array([0.005, 0, 0.001]) + 0.224
            for channel in range(input_data.shape[0]):
                input_data[channel, :, :] = (input_data[channel, :, :] / 255 - mean[channel]) / std[channel]
            return input_data

        image_names = os.listdir(images_folder)
        image_names.sort()
        if size_limit > 0 and len(image_names) >= size_limit:
            end_index = start_index + size_limit
            if end_index > len(image_names):
                end_index = len(image_names)
            batch_filenames = [image_names[i] for i in range(start_index, end_index)]
        else:
            batch_filenames = image_names

        unconcatenated_batch_data = []
        image_size_list = []

        for image_name in batch_filenames:
            image_filepath = images_folder + '/' + image_name
            img = Image.open(image_filepath)
            
            image_data = preprocess_images(img)
            image_data = np.expand_dims(image_data, 0)
            print(f'img.shape = {image_data.shape}')
            print(f'img.max = {image_data.max()}')
            print(f'img.min = {image_data.min()}')
            unconcatenated_batch_data.append(image_data)
            image_size_list.append(np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2))

        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data, batch_filenames, image_size_list

loader = ImageNetDataReader('../input_rotation/testA',width=1024,
                            height=1024,start_index=0,end_index=4,
                            stride=1,batch_size=1,
                            model_path='/home/ubuntu/transformer-distillation/width1.onnx',
                            input_name='input')
from onnx import shape_inference

nodes = ['/model/down_levels.0/down_levels.0.0/self_attn/norm/Add_1',
 '/model/down_levels.0/down_levels.0.0/self_attn/Add_1',
 '/model/down_levels.0/down_levels.0.0/self_attn/Add_2',
 '/model/down_levels.0/down_levels.0.0/ff/norm/Add_1',
 '/model/down_levels.0/down_levels.0.1/self_attn/norm/Add_1',
 '/model/down_levels.0/down_levels.0.1/self_attn/Add_1',
 '/model/down_levels.0/down_levels.0.1/self_attn/Add_2',
 '/model/down_levels.0/down_levels.0.1/ff/norm/Add_1',
 '/model/down_levels.1/down_levels.1.0/self_attn/norm/Add_1',
 '/model/down_levels.1/down_levels.1.0/self_attn/Add_1',
 '/model/down_levels.1/down_levels.1.0/self_attn/Add_2',
 '/model/down_levels.1/down_levels.1.0/ff/norm/Add_1',
 '/model/down_levels.1/down_levels.1.1/self_attn/norm/Add_1',
 '/model/down_levels.1/down_levels.1.1/self_attn/Add_1',
 '/model/down_levels.1/down_levels.1.1/self_attn/Add_2',
 '/model/down_levels.1/down_levels.1.1/ff/norm/Add_1',
 '/model/down_levels.2/down_levels.2.0/self_attn/norm/Add_1',
 '/model/down_levels.2/down_levels.2.0/self_attn/Add_1',
 '/model/down_levels.2/down_levels.2.0/self_attn/Add_2',
 '/model/down_levels.2/down_levels.2.0/ff/norm/Add_1',
 '/model/down_levels.2/down_levels.2.1/self_attn/norm/Add_1',
 '/model/down_levels.2/down_levels.2.1/self_attn/Add_1',
 '/model/down_levels.2/down_levels.2.1/self_attn/Add_2',
 '/model/down_levels.2/down_levels.2.1/ff/norm/Add_1',
 '/model/down_levels.3/down_levels.3.0/self_attn/norm/Add_1',
 '/model/down_levels.3/down_levels.3.0/self_attn/Add_1',
 '/model/down_levels.3/down_levels.3.0/self_attn/Add_2',
 '/model/down_levels.3/down_levels.3.0/ff/norm/Add_1',
 '/model/down_levels.3/down_levels.3.1/self_attn/norm/Add_1',
 '/model/down_levels.3/down_levels.3.1/self_attn/Add_1',
 '/model/down_levels.3/down_levels.3.1/self_attn/Add_2',
 '/model/down_levels.3/down_levels.3.1/ff/norm/Add_1',
 '/model/mid_level/mid_level.0/self_attn/norm/Add_1',
 '/model/mid_level/mid_level.0/self_attn/Add_1',
 '/model/mid_level/mid_level.0/self_attn/Add_2',
 '/model/mid_level/mid_level.0/ff/norm/Add_1',
 '/model/mid_level/mid_level.1/self_attn/norm/Add_1',
 '/model/mid_level/mid_level.1/self_attn/Add_1',
 '/model/mid_level/mid_level.1/self_attn/Add_2',
 '/model/mid_level/mid_level.1/ff/norm/Add_1',
 '/model/up_levels.3/up_levels.3.0/self_attn/norm/Add_1',
 '/model/up_levels.3/up_levels.3.0/self_attn/Add_1',
 '/model/up_levels.3/up_levels.3.0/self_attn/Add_2',
 '/model/up_levels.3/up_levels.3.0/ff/norm/Add_1',
 '/model/up_levels.3/up_levels.3.1/self_attn/norm/Add_1',
 '/model/up_levels.3/up_levels.3.1/self_attn/Add_1',
 '/model/up_levels.3/up_levels.3.1/self_attn/Add_2',
 '/model/up_levels.3/up_levels.3.1/ff/norm/Add_1',
 '/model/up_levels.2/up_levels.2.0/self_attn/norm/Add_1',
 '/model/up_levels.2/up_levels.2.0/self_attn/Add_1',
 '/model/up_levels.2/up_levels.2.0/self_attn/Add_2',
 '/model/up_levels.2/up_levels.2.0/ff/norm/Add_1',
 '/model/up_levels.2/up_levels.2.1/self_attn/norm/Add_1',
 '/model/up_levels.2/up_levels.2.1/self_attn/Add_1',
 '/model/up_levels.2/up_levels.2.1/self_attn/Add_2',
 '/model/up_levels.2/up_levels.2.1/ff/norm/Add_1',
 '/model/up_levels.1/up_levels.1.0/self_attn/norm/Add_1',
 '/model/up_levels.1/up_levels.1.0/self_attn/Add_1',
 '/model/up_levels.1/up_levels.1.0/self_attn/Add_2',
 '/model/up_levels.1/up_levels.1.0/ff/norm/Add_1',
 '/model/up_levels.1/up_levels.1.1/self_attn/norm/Add_1',
 '/model/up_levels.1/up_levels.1.1/self_attn/Add_1',
 '/model/up_levels.1/up_levels.1.1/self_attn/Add_2',
 '/model/up_levels.1/up_levels.1.1/ff/norm/Add_1',
 '/model/up_levels.0/up_levels.0.0/self_attn/norm/Add_1',
 '/model/up_levels.0/up_levels.0.0/self_attn/Add_1',
 '/model/up_levels.0/up_levels.0.0/self_attn/Add_2',
 '/model/up_levels.0/up_levels.0.0/ff/norm/Add_1',
 '/model/up_levels.0/up_levels.0.1/self_attn/norm/Add_1',
 '/model/up_levels.0/up_levels.0.1/self_attn/Add_1',
 '/model/up_levels.0/up_levels.0.1/self_attn/Add_2',
 '/model/up_levels.0/up_levels.0.1/ff/norm/Add_1',
 '/model/out_norm/Add']

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_static,quantize_dynamic, QuantType,QuantFormat
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)

    inferred_model = shape_inference.infer_shapes(onnx_opt_model)
    onnx.save(inferred_model, "inferred_model.onnx")
    import subprocess
    command = ["onnxsim", "inferred_model.onnx", "inferred_model.onnx"]
    command2 = ["python", "-m", "onnxruntime.quantization.preprocess", "inferred_model.onnx","inferred_model.onnx"]
    # Run the command
    _ = subprocess.run(command, capture_output=True, text=True)

    quantize_static("inferred_model.onnx",
                     quantized_model_path,
                     calibration_data_reader=loader,
                     activation_type=QuantType.QInt8,
                     weight_type=QuantType.QInt8,

                     extra_options={'ActivationSymmetric':True})

    print(f"quantized model saved to:{quantized_model_path}")
    print('ONNX full precision model size (MB):', os.path.getsize(onnx_model_path)/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(quantized_model_path)/(1024*1024))

quantize_onnx_model('/home/ubuntu/transformer-distillation/ps1-self-no-window.onnx', '/home/ubuntu/transformer-distillation/width2-q.onnx')