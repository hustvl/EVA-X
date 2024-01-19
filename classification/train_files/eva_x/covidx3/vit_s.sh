CKPT_DIR='pretrained/eva_x_small_patch16_merged520k_mim.pt'
DATASET_DIR='datasets/covidx3'
train_list='datasets/data_splits/covidx3/train_COVIDx9A.txt'
test_list='datasets/data_splits/covidx3/test_COVIDx9A.txt'

for DATA_PCT in 1 0.1 0.01
do
    SAVE_DIR=./output/covidx3/vit_s_eva_x_covidx3_${DATA_PCT}data
    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env train.py \
        --finetune ${CKPT_DIR} \
        --output_dir ${SAVE_DIR} \
        --log_dir ${SAVE_DIR} \
        --input_size 480 \
        --batch_size 128 \
        --checkpoint_type "" \
        --epochs 10 \
        --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
        --fixed_lr \
        --model 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
        --warmup_epochs 1 \
        --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
        --data_path ${DATASET_DIR} \
        --num_workers 16 \
        --train_list ${train_list} \
        --val_list './' \
        --test_list ${test_list} \
        --dataset covidx \
        --nb_classes 3 \
        --eval_interval 1 \
        --use_mean_pooling \
        --data_pct ${DATA_PCT}
done