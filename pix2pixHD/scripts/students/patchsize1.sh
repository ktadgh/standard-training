<<<<<<< Updated upstream
python train.py --name teacher-vgg --config_path /home/ubuntu/tdist-extra/k-diffusion-onnx/configs/mini_config_oxford_flowers_shifted_window.json --alpha 1e-09 --experiment_name teacher-vgg --resume_distill_epoch 0 --niter 100 --niter_decay 100 --save_epoch_freq 5 --dataroot "../../transformer-distillation/images/images" --batchSize 24 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 256 --teacher_vgg
=======
python train.py --name patchsize1-naivegan-layer-end-higher --config_path /home/ubuntu/tdist-extra/k-diffusion-onnx/configs/mini_config_oxford_flowers_shifted_window.json --alpha 1.678e-6 --experiment_name patchsize1-naivegan-layer-end-higher --resume_distill_epoch 0 --niter 100 --niter_decay 100 --save_epoch_freq 5 --dataroot "../../transformer-distillation/images/images" --batchSize 24 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 256
>>>>>>> Stashed changes
