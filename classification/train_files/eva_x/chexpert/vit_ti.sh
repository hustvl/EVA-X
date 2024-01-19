DATASET_DIR='datasets/chexpert/'
CKPT_DIR='pretrained/eva_x_tiny_patch16_merged520k_mim.pt'
SAVE_DIR='./output/chexpert/vit_ti_eva_x_chexpert'
TRAIN_LIST='datasets/data_splits/chexpert/train.csv'
VAL_LIST='./'                                                                                           # not used
TEST_LIST='datasets/data_splits/chexpert/valid.csv'

# fake epoch
# true epoch == fake epoch // 20
# We train the special dataset with special ways.

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env train.py \
    --dataset chexpert \
    --input_size 224 \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --checkpoint_type "" \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --fixed_lr \
    --model 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
    --warmup_epochs 0 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 8 \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --nb_classes 5 \
    --eval_interval 1 \
    --use_mean_pooling \
    --stop_grad_conv1 \
    --use_smooth_label
