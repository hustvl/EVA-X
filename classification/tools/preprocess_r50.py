"""
We use this file to convert ResNet-50 pretrained weights
to meet our need.
"""
import torch
import datetime
from collections import OrderedDict

def print_with_prefix(msg):
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f'[{time} EVA-X ]:'
    print(f'{prefix} {msg}')

def convert_medklip(
    src_dir='/data/jingfengyao/pretrained_models/EVA-X-competition/medklip_checkpoint_final.pth',
    tgt_dir='r50_medklip.pth',
):
    """
    Download MedKLIP pretrained weights from:
    https://github.com/MediaBrain-SJTU/MedKLIP
    """
    # Load the weight from the source directory
    weights = torch.load(src_dir)['model']
    
    print_with_prefix(f'Pretrained weights loaded from {src_dir}')

    new_weights = OrderedDict()
    for k, v in weights.items():
        if 'res_features' in k:
            new_k = k.replace('module.res_features.0', 'conv1')
            new_k = new_k.replace('module.res_features.1', 'bn1')
            new_k = new_k.replace('module.res_features.2', 'relu')
            new_k = new_k.replace('module.res_features.3', 'maxpool')
            new_k = new_k.replace('module.res_features.4', 'layer1')
            new_k = new_k.replace('module.res_features.5', 'layer2')
            new_k = new_k.replace('module.res_features.6', 'layer3')
            new_weights[new_k] = v
            
            print_with_prefix(f'converted {k} to {new_k}')

    # Save the converted weight to the target directory
    torch.save(new_weights, tgt_dir)
    print_with_prefix(f'Pretrained weights saved to {tgt_dir}')

def convert_mgca(
    src_dir='/home/jingfengyao/code/medical/medical_mae/pretrained/mgca_resnet_50.ckpt',
    tgt_dir='r50_mgca.pth',
):
    """
    Download MGCA pretrained weights from:
    https://github.com/HKU-MedAI/MGCA
    """
    # Load the weight from the source directory
    weights = torch.load(src_dir)['state_dict']
    
    print_with_prefix(f'Pretrained weights loaded from {src_dir}')
    new_weights = OrderedDict()
    for k, v in weights.items():
        if 'img_encoder_q.model.' in k:
            new_k = k.replace('img_encoder_q.model.', '')
            new_weights[new_k] = v
            
            print_with_prefix(f'converted {k} to {new_k}')

    # Save the converted weight to the target directory
    torch.save(new_weights, tgt_dir)
    print_with_prefix(f'Pretrained weights saved to {tgt_dir}')

def convert_biovil(
    src_dir='/home/jingfengyao/code/medical/medical_mae/pretrained/biovil_image_resnet50_proj_size_128.pt',
    tgt_dir='r50_biovil.pth',
):
    """
    Download BioVIL pretrained weights from:
    https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/tree/main
    """
    # Load the weight from the source directory
    weights = torch.load(src_dir)

    print_with_prefix(f'Pretrained weights loaded from {src_dir}')
    new_weights = OrderedDict()
    for k, v in weights.items():
        if 'encoder.encoder.' in k:
            if 'fc' not in k:
                new_k = k.replace('encoder.encoder.', '')
                new_weights[new_k] = v
                
                print_with_prefix(f'converted {k} to {new_k}')
            elif 'fc' in k:
                pass
    # Save the converted weight to the target directory
    torch.save(new_weights, tgt_dir)
    print_with_prefix(f'Pretrained weights saved to {tgt_dir}')

def convert_mocov2(
    src_dir='/home/jingfengyao/code/medical/EVA-X/reproduce/pretrained/resnet50_imagenet_mocov2.pth',
    tgt_dir='r50_mocov2.pth',
):
    """
    Download MoCoV2 pretrained weights from:
    https://drive.google.com/file/d/1GVSc3TOEhItliMToyY8Z4oHW_Kxf8cRj/view?usp=share_link
    """
    print_with_prefix(f'Pretrained weights loaded from {src_dir}')
    # Load the weight from the source directory
    weights = torch.load(src_dir)

    # nothing to do
    new_weights = OrderedDict()
    new_weights = weights

    # Save the converted weight to the target directory
    torch.save(new_weights, tgt_dir)
    print_with_prefix(f'Pretrained weights saved to {tgt_dir}')



if __name__ == "__main__":
    tgt_root = 'pretrained'
    convert_medklip(tgt_dir=f'{tgt_root}/r50_medklip.pth')
    convert_mgca(tgt_dir=f'{tgt_root}/r50_mgca.pth')
    convert_biovil(tgt_dir=f'{tgt_root}/r50_biovil.pth')
    convert_mocov2(tgt_dir=f'{tgt_root}/r50_mocov2.pth')