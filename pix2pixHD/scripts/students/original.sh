python train.py --name ps8-flattened --resume_distill_epoch 0 --config_path /home/ubuntu/tdist-flat/configs/hdit-shifted-windows/patchsize1.json --experiment_name ps8-flattened --save_epoch_freq 1 --niter 100 --niter_decay 100 --dataroot "../../images" --batchSize 3 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024