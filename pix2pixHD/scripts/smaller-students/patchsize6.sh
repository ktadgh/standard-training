python train_6.py --name patchsize6 --resume_distill_epoch 0 --config_path /home/ubuntu/tdist-9ch/configs/hdit-shifted-windows/patchsize6.json --experiment_name patchsize6 --niter 100 --niter_decay 100 --save_epoch_freq 1  --dataroot "../../transformer-distillation/images/images" --batchSize 8 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024 --alpha 1e-06 --alpha1 0.31 --alpha2 0.31 --alpha5 0.36 --teacher_adv --teacher_feat --teacher_vgg --aim_repo /home/ubuntu/aim/1024-pix2pixHD-runs