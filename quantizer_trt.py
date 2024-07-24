# Import Python Standard Library dependencies
import json
import os
from pathlib import Path
import random

# Import utility functions
# from cjm_psl_utils.core import download_file, file_extract
# from cjm_pil_utils.core import resize_img, get_img_files
from PIL import Image
# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Do not truncate the contents of cells and display all rows and columns
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

# Import PIL for image manipulation
from PIL import Image

# Import ONNX dependencies
import onnxruntime as ort # Import the ONNX Runtime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, create_calibrator, write_calibration_table


onnx_file_path = '/home/ubuntu/transformer-distillation/ps1-self-no-window.onnx'
sample_img_paths = ['../samples/img_5099.png','../samples/img_5098.png']
trt_cache_dir = 'cache'



class CalibrationDataReaderCV(CalibrationDataReader):
    """
    A subclass of CalibrationDataReader specifically designed for handling
    image data for calibration in computer vision tasks. This reader loads,
    preprocesses, and provides images for model calibration.
    """
    
    def __init__(self, img_file_paths, target_sz, input_name='input'):
        """
        Initializes a new instance of the CalibrationDataReaderCV class.
        
        Args:
            img_file_paths (list): A list of image file paths.
            target_sz (tuple): The target size (width, height) to resize images to.
            input_name (str, optional): The name of the input node in the ONNX model. Default is 'input'.
        """
        super().__init__()  # Initialize the base class
        
        # Initialization of instance variables
        self._img_file_paths = img_file_paths
        self.input_name = input_name
        self.enum = iter(img_file_paths)  # Create an iterator over the image paths
        self.target_sz = target_sz
        
    def get_next(self):
        """
        Retrieves, processes, and returns the next image in the sequence as a NumPy array suitable for model input.
        
        Returns:
            dict: A dictionary with a single key-value pair where the key is `input_name` and the value is the
                  preprocessed image as a NumPy array, or None if there are no more images.
        """
        
        img_path = next(self.enum, None)  # Get the next image path
        if not img_path:
            return None  # If there are no more paths, return None

        # Load the image from the filepath and convert to RGB
        image = Image.open(img_path).convert('RGB')

        # Resize the image to the target size
        input_img = image #resize_img(image, target_sz=self.target_sz, divisor=1)
        
        # Convert the image to a NumPy array, normalize, and add a batch dimension
        input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None] / 255

        # Return the image in a dictionary under the specified input name
        return {self.input_name: input_tensor_np}


# Save path for temporary ONNX model used during calibration process
augmented_model_path = onnx_file_path.replace('.onnx', '') + '-augmented.onnx'

try:
    # Create a calibrator object for the ONNX model.
    calibrator = create_calibrator(
        model=onnx_file_path, 
        op_types_to_calibrate=None, 
        augmented_model_path=augmented_model_path, 
        calibrate_method=CalibrationMethod.MinMax
    )

    # Set the execution providers for the calibrator.
    calibrator.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Initialize the custom CalibrationDataReader object
    calibration_data_reader = CalibrationDataReaderCV(img_file_paths=sample_img_paths, 
                                                      target_sz=512, 
                                                      input_name=calibrator.model.graph.input[0].name)

    # Collect calibration data using the specified data reader.
    calibrator.collect_data(data_reader=calibration_data_reader)

    # Initialize an empty dictionary to hold the new compute range values.
    new_compute_range = {}

    # Compute data and update the compute range for each key in the calibrator's data.
    for k, v in calibrator.compute_data().data.items():
        # Extract the min and max values from the range_value.
        v1, v2 = v.range_value
        # Convert the min and max values to float and store them in the new_compute_range dictionary.
        new_compute_range[k] = (float(v1.item()), float(v2.item()))
        
    # Write the computed calibration table to the specified directory.
    write_calibration_table(new_compute_range, dir=str(trt_cache_dir))
    
except Exception as e:
    # Catch any exceptions that occur during the calibration process.
    print("An error occurred:", e)

# finally:
#     Remove temporary ONNX file created during the calibration process
#     if augmented_model_path.exists():
#         augmented_model_path.unlink()


import subprocess
import onnx


onnx_file_path = 'ps1-self-no-window.onnx'
from onnx import shape_inference

# Load the ONNX model
model_path ='ps1-self-no-window.onnx'
model = onnx.load(model_path)

# Perform shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the inferred model (optional)
onnx.save(inferred_model, 'ps1-self-no-window-nothing.onnx')


onnx_file_path = 'ps1-self-no-window-nothing.onnx'
import os 


providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0, # The device ID
        'trt_max_workspace_size': 46e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
        'trt_engine_cache_enable': False, # Enable TensorRT engine caching
        'trt_int8_enable': True, # Enable INT8 mode in TensorRT
        'trt_int64_enable': True, # Enable INT8 mode in TensorRT
        'trt_engine_cache_enable':False,
        'trt_context_memory_sharing_enable':True,
        'trt_auxiliary_streams':0, # will give optimal memory usage
        'trt_int8_calibration_table_name': 'calibration.flatbuffers', # INT8 calibration table file for non-QDQ models in INT8 mode
    })
]
ort.set_default_logger_severity(1)
sess_opt = ort.SessionOptions()
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_opt.enable_mem_pattern = True
sess_opt.use_deterministic_compute = True
sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_opt.enable_cpu_mem_arena = True
sess_opt.log_severity_level = 1
sess_opt.intra_op_num_threads = 1

# Load the model and create an InferenceSession

# model = onnx.load(onnx_file_path)
# for tensor in model.graph.input:
#     print(tensor.name, tensor.type.tensor_type.elem_type)
# raise ValueError()

session = ort.InferenceSession(onnx_file_path, sess_options=sess_opt, providers=providers)

x = np.random.rand(1,3,512,512).astype(np.float32)
import time
for _ in range(50):
    t0 = time.time()
    (session.run(None, {"input": x}))
    t1 = time.time()
    print(f'elapsed = {t1 -t0}')


print(f'Session worked OK')