python train.py --name width2 --save_epoch_freq 5 --config_path /home/ubuntu/transformer-distillation/configs/hdit-shifted-windows/width2.json --experiment_name width2 --niter 100 --niter_decay 100 --dataroot "../../pix2pix_train_val_test" --batchSize 4 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024 