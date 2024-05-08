<div align="center">

#  <img src="figs/logo.png" height="28"> *EVA-X*

## *E*xploring Generalizable *V*isual Representation with *A*dvanced Self-supervised Learning for *X*-Ray Images </h2>

[Jingfeng Yao](https://github.com/JingfengYao)<sup>1</sup>, [Xinggang Wang](https://scholar.google.com/citations?user=qNCTLV0AAAAJ&hl=zh-CN)<sup>1 📧</sup>, [Yuehao Song]()<sup>1</sup>, [Huangxuan Zhao]()<sup>2</sup>, [Jun Ma]()<sup>3,4,5</sup>, [Wenyu Liu]()<sup>1</sup>, [Bo Wang]()<sup>3,4,5,6,7</sup>

<sup>1</sup>School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, Hubei, China

<sup>2</sup>Department of Radiology, Union Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, Hubei, China

<sup>3</sup>Peter Munk Cardiac Centre, University Health Network, Toronto, Ontario, Canada

<sup>4</sup>Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Ontario, Canada

<sup>5</sup>Vector Institute for Artificial Intelligence, Toronto, Ontario, Canada

<sup>6</sup>AI Hub, University Health Network, Toronto, Ontario, Canada

<sup>7</sup>Department of Computer Science, University of Toronto, Toronto, Ontario, Canada

(<sup>📧</sup>) corresponding author.

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![authors](https://img.shields.io/badge/by-hustvl-green)](https://github.com/hustvl)

</div>

## News

* **`May xxth, 2024`:**  EVA-X is released!

## Introduction

**EVA-X are Vision Transformers (ViTs) pre-trained especially for medical X-ray images.**

**Take them away for your own X-ray tasks!**

<p align="center">
    <img src="figs/pipeline.png" alt="Overall Architecture" width="800">
</p>
<p align="center">
    <img src="figs/poster.png" alt="Poster" width="800">
</p>

| EVA-X Series | Architecture | #Params | Checkpoint | Tokenizer |  MIM epochs |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|
| <img src="figs/logo.png" height="17"> EVA-X-Ti    |  ViT-Ti/16   | 6M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt)| MGCA-ViT-B/16 | 900 |
| <img src="figs/logo.png" height="17"> EVA-X-S    |  ViT-S/16   | 22M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt)|MGCA-ViT-B/16 | 600 |
| <img src="figs/logo.png" height="17"> EVA-X-B    |  ViT-B/16   | 86M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_base_patch16_merged520k_mim.pt)|MGCA-ViT-B/16 | 600 |

## Contents

EVA-X has released all experimental code from the paper. Here is our contents; please refer to the corresponding sections as needed.

Use EVA-X as your backbone:
- [Quick Start](#quick-start)

Pre-training and Finetuning:
- [X-ray Image Classification](#classification)
- [X-ray Image Segmentation](segmentation)

Interpretability analysis:
- [Grad-CAM Analysis](grad_cam)

## Quick Start

Launch EVA-X with three simple steps. Try EVA-X representations for your own X-rays!

1. Download pre-trained weights.

2. Install [pytorch_image_models](https://github.com/huggingface/pytorch-image-models)
    ```
    ! pip install timm==0.9.0
    ```

3. Initialize EVA-X with simple python codes. You could also check ``eva_x.py`` to modify it for your own X-ray tasks.
    ```
    from eva_x import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16

    eva_pt = '/path/to/eva_s_16.pt'
    model = eva_x_small_patch16(pretrained=/path/to/pre-trained)
    ```

## Classification
### Data Preparation

We have done the most of the work before. You could use the datasets with simple steps.

1. Download [Chest X-Ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) / [CheXpert](https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small) / [CovidX-CXR-3](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2/versions/5) /  [CovidX-CXR-4](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) from the given links. (Note: for CovidX dataset, we use the version-5 and version-9.)

2. Prepare them by running the following codes. It will prepare and check the datasets automatically.
   ```
   python datasets/prepare_dataset.py
   ```
If there is no error, it's done!
### Weights Preparation

#### Run EVA-X Only

If you just want to use EVA-X for medical image classification, you only need to download the [weights for EVA-X](#introduction).


#### Reproduce all methods

In our experiments, in addition to EVA-X, we *used 15 different visual representations for comparison*. You can choose to download the weights you need.

| Architecture | #Params | Methods / Downloads |
|:------------|:-------|:----------|
 ResNet50 | 24M |[ImageNet](https://download.pytorch.org/models/resnet50-19c8e357.pth); [MoCov2](https://drive.google.com/file/d/1GVSc3TOEhItliMToyY8Z4oHW_Kxf8cRj/view?usp=share_link); [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP) [MGCA](https://github.com/HKU-MedAI/MGCA); [BioViL](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/tree/main)|
| DenseNet121 | 8M | [ImageNet](https://download.pytorch.org/models/densenet121-a639ec97.pth);  [MoCov2](https://drive.google.com/file/d/1idLcwL4C0eSGoLc5PI4ZWHzNLq_mJY-g/view?usp=share_link); [Medical MAE](https://drive.google.com/file/d/1f5KePD48QmHua7C5HBUV3i0PqNPkgmqj/view?usp=share_link) |
| ViT-S/16 | 22M |[DEiT](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth); [MAE](https://drive.google.com/file/d/1QeAIWJWuNcccF09502xcnQ_2tbc2jRQG/view?usp=share_link); [EVA-02](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_S_pt_in21k_p14.pt); [Medical MAE](https://drive.google.com/file/d/1Yok1RemqP27iKJ5BUuoHhLRBB1GycITx/view?usp=share_link)|

Some weights should be processed for training. We provide codes for them:

```
# process ResNet50s
python tools/preprocess_r50.py

# process EVA02
python tools/interpolate14to16.py
```
## Run

### Run EVA-X Only

We use 4 RTX 3090 GPUs for finetuning. We have fixed most of the randomness. So the results are reproducible. We have released our weights, you can also eval with them directly ([how to eval?](#evaluate) ) . Run the following codes to finetune EVA-X in classification datasets:

***Chest X-Ray14***

The reported mAUC is evaluated on the *official test set*.

```
sh train_files/eva_x/cxr14/vit_ti.sh
sh train_files/eva_x/cxr14/vit_s.sh
sh train_files/eva_x/cxr14/vit_s.sh
```

| EVA-X Series | Architecture | #Params |  MIM epochs | FT configs | FT epochs | Res | mAUC |Checkpoint |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-Ti    |  ViT-Ti/16   | 6M     | 900 | [config](train_files/eva_x/cxr14/vit_ti.sh) | 45  | 224x224  | 82.4 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim_cxr14_ft.pth) |
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-S    |  ViT-S/16   | 22M     | 600 | [config](train_files/eva_x/cxr14/vit_s.sh) | 30  | 224x224  | 83.3 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_cxr14_ft.pth) |
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-B    |  ViT-B/16   | 86M     | 600 | [config](train_files/eva_x/cxr14/vit_b.sh) | 10  | 224x224  | 83.5 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_base_patch16_merged520k_mim_cxr14_ft.pth) |

***CheXpert***

The reported mAUC is evaluated on the *official val set*.

```
sh train_files/eva_x/chexpert/vit_ti.sh
sh train_files/eva_x/chexpert/vit_s.sh
```

| EVA-X Series | Architecture | #Params |  MIM epochs | FT configs | FT epochs | Res | mAUC |Checkpoint |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-Ti    |  ViT-Ti/16   | 6M     | 900 | [config](train_files/eva_x/chexpert/vit_ti.sh) | 5  | 224x224  | 89.6 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_mrerged520k_mim_chexpert_ft.pth) |
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-S    |  ViT-S/16   | 22M     | 600 | [config](train_files/eva_x/chexpert/vit_s.sh) | 5  | 224x224  | 90.1 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_chexpert_ft.pth) |

***CovidX-CXR-3&4***

We train EVA-X with these two dataset with 1%, 10% and 100% X-ray images of the training set. You could change the data percentage in the provided training files.

The reported mAcc is evaluated *on the official test/val set*.
```
sh train_files/eva_x/covidx3/vit_s.sh
sh train_files/eva_x/covidx4/vit_s.sh
```

| EVA-X Series | Architecture | #Params |  Dataset | FT configs | FT epochs | Res | mAcc |Checkpoint |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-S    |  ViT-S/16   | 22M     | CovidX-CXR-3 | [config](train_files/eva_x/covidx3/vit_s.sh) | 10  | 480x480  | 97.5 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_covidx3_ft.pth) |
| <img src="figs/x-ray-logo.png" width="18"> EVA-X-S    |  ViT-S/16   | 22M     | CovidX-CXR-4 | [config](train_files/eva_x/covidx4/vit_s.sh) | 10  | 480x480  | 96.0 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim_covidx4_ft.pth) |

*Note*: *We do not claim that any of the above models can be used directly for disease diagnosis. Please seek professional medical help when needed.*

### Run with other methods

We compare EVA-X with 15 different visual representations. We compare them on three datasets: Chest X-Ray14, CovidX-CXR-3, CovidX-CXR-4. We have uploaded part of their finetuned weights for analysis. More results please refer to our paper. **We appreciate all of their excellent research work!**

***Chest X-Ray14***

For example, run:
```
# choose the file you want to reproduce
sh train_files/resnet/cxr14/r50_biovil.sh
```

| PT Methods | Architecture | #Params | FT configs | FT epochs | Res | mAUC |Checkpoint |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [MoCov2](https://arxiv.org/abs/2003.04297)    |  DN121   | 7M  | [config](train_files/dn121/cxr14/dn121_mocov2_mae_cxr14.sh) | 75  | 224x224  | 80.8 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/densenet121_mocov2_pt_cxr14_ft.pth) |
| [Medical MAE](https://arxiv.org/abs/2210.12843)    |  DN121   | 7M  | [config](train_files/dn121/cxr14/dn121_medical_mae_cxr14.sh) | 75  | 224x224  | 81.3 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/densenet121_medical_mae_pt_cxr14_ft.pth) |
| [BioViL](https://arxiv.org/abs/2204.09817)    |  ResNet50   | 24M  | [config](train_files/resnet/cxr14/r50_biovil.sh) | 75  | 224x224  | 80.8 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/resnet50_biovil_pt_cxr14_ft.pth) |
| [MGCA](https://arxiv.org/abs/2210.06044)    |  ResNet50   | 24M  | [config](train_files/resnet/cxr14/r50_mgca.sh) | 75  | 224x224  | 81.3 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/resnet50_mgca_pt_cxr14_ft.pth) |
| [MedKLIP](https://arxiv.org/abs/2301.02228)    |  ResNet50   | 24M  | [config](train_files/resnet/cxr14/r50_medklip.sh) | 75  | 224x224  | 81.2 |[🤗download](https://huggingface.co/MapleF/eva_x/blob/main/resnet50_medklip_pt_cxr14_ft.pth) |
| [Medical MAE](https://arxiv.org/abs/2210.12843)    |  ViT-S   | 22M  | [config](train_files/vit/cxr14/vit_s_medical_mae_cxr14.sh) | 75  | 224x224  | 82.3 | [official download](https://drive.google.com/file/d/1DkZMkXcFpj_SdffYZzw-Dq5clfo_YqjZ/view?usp=share_link) |

### Evaluate

You could eval with our finetuned models by slightly changing the train files.

1. Download our pre-trained CKPT above.
2. Change the train file:
   ```
   (1) change weights
   CKPT_PATH='/path/to/mim/weights' --> '/path/to/finetuned/weights'
   (2) add '--eval' to the end of args
   ```

## Acknowledgements

Our codes are built upon [EVA](https://github.com/baaivision/EVA/tree/master/EVA-01), [EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02), [MGCA](https://github.com/HKU-MedAI/MGCA), [Medical MAE](https://github.com/lambert-x/medical_mae), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/huggingface/pytorch-image-models), [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch), [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam) Thansk for these great repos!
