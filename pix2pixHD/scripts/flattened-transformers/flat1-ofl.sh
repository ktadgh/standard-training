for i in 0 7; do
    python train_no_accum.py --cutoff $i --alpha_temporal 1.0 --name flat-og-32-max-4-cutoff-alpha1-noaccum$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha1-noaccum$i
done


for i in 0 7; do
    python train.py --cutoff $i --alpha_temporal 1.0 --name flat-og-32-max-4-cutoff-alpha1-outplace$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 1 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha1-cutoff-outplace$i
done


for i in 0 7; do
    python train_no_accum.py --cutoff $i --alpha_temporal 0.0001 --name flat-og-32-max-4-cutoff-alpha1e-4-noaccum-cutoff-$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha1e-4-noaccum-cutoff-$i
done

for i in 0 7; do
    python train.py --cutoff $i --alpha_temporal 0.0001 --name flat-og-32-max-4-cutoff-alpha1e-4-cutoff-$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha1e-4-cutoff$i
done


for i in 0; do
    python train_no_accum.py --cutoff $i --alpha_temporal 0.0 --name flat-og-32-max-4-cutoff-alpha0-noaccum-cutoff-$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha0-noaccum-cutoff-$i
done

for i in 0; do
    python train.py --cutoff $i --alpha_temporal 0.0 --name flat-og-32-max-4-cutoff-alpha0-cutoff-$i \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
    --experiment_name flat-og-32-max-4-cutoff-alpha0-cutoff$i
done