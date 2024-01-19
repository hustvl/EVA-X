DATASET_DIR='datasets/cxr14'
# CKPT_DIR='pretrained/r50_medklip.pth'
CKPT_DIR='pretrained/resnet50_medklip_pt_cxr14_ft.pth'
SAVE_DIR='./output/cxr14/r50_medklip_cxr14'
TRAIN_LIST='datasets/data_splits/cxr14/train_official.txt'
VAL_LIST='datasets/data_splits/cxr14/val_official.txt'         # not used
TEST_LIST='datasets/data_splits/cxr14/test_official.txt'

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env train.py \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --checkpoint_type "" \
    --epochs 75 \
    --blr 2.5e-4 --weight_decay 0.05 \
    --model 'resnet50' \
    --warmup_epochs 5 \
    --drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 8 \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --nb_classes 14 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --eval