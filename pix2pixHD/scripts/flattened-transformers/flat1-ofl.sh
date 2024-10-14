
for i in 0; do
    python train.py --cutoff 0 --alpha_temporal 0.0 --name prop-alpha-0 \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 1 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/patchsize8-32.json \
    --experiment_name prop-alpha-0
done

for i in 0; do
    python train.py --cutoff 0 --alpha_temporal 1.0 --name prop-alpha-1 \
    --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
    --save_epoch_freq 2 --dataroot "../../images" --accum_iter 1 --batchSize 3 --label_nc 0 \
    --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/patchsize8-32.json \
    --experiment_name prop-alpha-1
done
# for i in 0 7; do
#     python train_no_accum.py --cutoff $i --alpha_temporal 1.0 --name no-accum-alpha1-cutoff-$i \
#     --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
#     --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
#     --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
#     --experiment_name no-accum-alpha1-cutoff-$i
# done


# for i in 0 7; do
#     python train.py --cutoff $i --alpha_temporal 1.0 --name accum-alpha1-cutoff-$i \
#     --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
#     --save_epoch_freq 2 --dataroot "../../images" --accum_iter 1 --batchSize 3 --label_nc 0 \
#     --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
#     --experiment_name accum-alpha1-cutoff-$i
# done


# for i in 0 7; do
#     python train_no_accum.py --cutoff $i --alpha_temporal 0.0001 --name no-accum-alpha1-e4-cutoff-$i \
#     --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
#     --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
#     --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
#     --experiment_name no-accum-alpha1-e4-cutoff-$i
# done

# for i in 0 7; do
#     python train.py --cutoff $i --alpha_temporal 0.0001 --name accum-alpha1-e4-cutoff-$i \
#     --no_flip --nThreads 12 --resume_distill_epoch 0 --alpha 1e-14 --niter 1 --niter_decay 0 \
#     --save_epoch_freq 2 --dataroot "../../images" --accum_iter 2 --batchSize 3 --label_nc 0 \
#     --no_instance --gpu_ids 0 --loadSize 1024 --config_path /home/ubuntu/tdist-flat/configs/flat-transformers/og-max-32.json \
#     --experiment_name accum-alpha1-e4-cutoff-$i
# done
