/home/ubuntu/tdist-flat/tdist/bin/python train_standard.py --name flat-standard-4-noskip-check --accum_iter 1 --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 100 --niter_decay 100 --save_epoch_freq 1 --dataroot "../../images" --batchSize 4 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json --experiment_name flat-standard-4-noskip-check