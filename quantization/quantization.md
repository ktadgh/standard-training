
# Explicit quantizing with onnxruntime and TensorRT
The first thing I tried was explicitly quantizing using onnxruntime. This turned out to be slower than even fp32 inference. It's possible that some operations were falling back to CUDA, I'm not sure as I didn't profile it. Explicit quantizing can also be done in pytorch - it involves inserting quantization and dequantization layers between the network layers. It can be done in manually specified places but I applied it everywhere.


# Implicit quantizing with tensorrt
The alternative is implicit quantization with tensorrt, in which tensorrt builds an engine based on whatever is fastest, int8 or fp32 etc. To do this I needed to use onnxruntime to generate the calibration configuration, which specitfies how to quantize (scale and zero-point) by running the network on some input images. 
- When using constant folding I got a fused operation error in onnxruntime, it dissapeared when using constant_folding=False on torch's onnx export.
- I had to disable all optimization for it to work in onnxruntime.
- After the configuration is done in onnxruntime you can pass it to trtexec using the --config flag. It has to be done using Entropy rather than MinMax as that's what tensorrt uses by default.
- However constant folding should be used for both configuration and inference so that the scales and zero-points match to the correct layer names.

