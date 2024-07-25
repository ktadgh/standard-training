import onnxruntime as ort
import torch
import numpy as np

input=torch.rand(1,3,1024,1024)
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0, # The device ID
        'trt_max_workspace_size': 24e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
        'trt_engine_cache_enable': False, # Enable TensorRT engine caching
        # 'trt_engine_cache_path': str(trt_cache_dir), # Path for TensorRT engine, profile files, and int8 calibration table
        'trt_int8_enable': True, # Enable int8 mode in TensorRT
        'trt_fp16_enable': True, 

    }),('CUDAExecutionProvider')
]

# providers = [
#     ('CUDAExecutionProvider')]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_mem_pattern = True
sess_options.use_deterministic_compute = True
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.enable_cpu_mem_arena = False



session = ort.InferenceSession("/home/ubuntu/transformer-distillation/width2-q.onnx",sess_options,providers=providers)

# Run inference with ONNX Runtime
input_data = input.cpu().numpy().astype(np.float32)
ort_inputs = {session.get_inputs()[0].name: input_data}
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(50):
    print(f'starting')
    start.record()
    onnx_output = session.run(None, ort_inputs)
    end.record()
    print(start.elapsed_time(end))
# Compare the outputs