python train.py --name width1 --config_path /home/ubuntu/transformer-distillation/configs/hdit-shifted-windows/width1.json --experiment_name width1 --niter 100 --niter_decay 100 --save_epoch_freq 5  --dataroot "../../pix2pix_train_val_test" --batchSize 1 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024 