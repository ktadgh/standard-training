python train.py --alpha_temporal 10.0 --name flat-og-32-max-4-at10 --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 100 --niter_decay 100 --save_epoch_freq 1 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json --experiment_name flat-og-32-max-4-at10