python get_cams.py \
    --model densenet121 \
    --ckpt pretrained/densenet121_mocov2_pt_cxr14_ft.pth \
    --save-path ./output/densenet121_mocov2_pt_cxr14_ft \
    --searching


python get_cams.py \
    --model densenet121 \
    --ckpt pretrained/densenet121_medical_mae_pt_cxr14_ft.pth \
    --save-path ./output/densenet121_medical_mae_pt_cxr14_ft \
    --searching

python get_cams.py \
    --model resnet50 \
    --ckpt pretrained/resnet50_biovil_pt_cxr14_ft.pth \
    --save-path ./output/resnet50_biovil_pt_cxr14_ft \
    --searching

python get_cams.py \
    --model resnet50 \
    --ckpt pretrained/resnet50_mgca_pt_cxr14_ft.pth \
    --save-path ./output/resnet50_mgca_pt_cxr14_ft \
    --searching

python get_cams.py \
    --model resnet50 \
    --ckpt pretrained/resnet50_medklip_pt_cxr14_ft.pth \
    --save-path ./output/resnet50_medklip_pt_cxr14_ft \
    --searching

python get_cams.py \
    --model vit_small_patch16 \
    --ckpt pretrained/vit-s_CXR_0.3M_mae.pth \
    --save-path ./output/vit_grad_cam \
    --searching

python get_cams.py \
    --model eva_x_small \
    --ckpt pretrained/eva_x_small_patch16_merged520k_mim_cxr14_ft.pth \
    --save-path ./output/eva_grad_cam \
    --searching