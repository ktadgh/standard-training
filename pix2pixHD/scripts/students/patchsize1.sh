python train.py --name patchsize1-naivegan --config_path /home/ubuntu/transformer-distillation/configs/hdit-shifted-windows/patchsize1.json --experiment_name patchsize1-naivegan --niter 100 --niter_decay 100 --save_epoch_freq 5 --dataroot "../../input_rotation_1024" --batchSize 4 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024