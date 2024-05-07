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
    <img src="figs/pipeline.png" alt="Overall Architecture" height="200">
    &nbsp;
    <img src="figs/poster.png" alt="Poster" height="200">
</p>

| EVA-X Series | Architecture | #Params | Checkpoint | Tokenizer |  MIM epochs |
|:------------:|:------------:|:-------:|:----------:|:----------:|:----------:|
| <img src="figs/logo.png" height="17"> EVA-X-Ti    |  ViT-Ti/16   | 6M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt)| MGCA-ViT-B/16 | 900 |
| <img src="figs/logo.png" height="17"> EVA-X-S    |  ViT-S/16   | 22M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt)|MGCA-ViT-B/16 | 600 |
| <img src="figs/logo.png" height="17"> EVA-X-B    |  ViT-B/16   | 86M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_base_patch16_merged520k_mim.pt)|MGCA-ViT-B/16 | 600 |

## Contents

EVA-X has released all experimental code from the paper. Here is our contents; please refer to the corresponding sections as needed.

Use EVA-X as your backbone:
- [Qucik Start](#qucik-start)

Pre-training and Finetuning:
- [MIM Pre-training with Medical CLIP](pretrain)
- [X-ray Image Classification](classification)
- [X-ray Image Segmentation](segmentation)

Interpretability analysis:
- [Grad-CAM Analysis](grad_cam)

## Qucik Start

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

## Acknowledgements

Our codes are built upon [EVA](https://github.com/baaivision/EVA/tree/master/EVA-01), [EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02), [MGCA](https://github.com/HKU-MedAI/MGCA), [Medical MAE](https://github.com/lambert-x/medical_mae), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/huggingface/pytorch-image-models), [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch), [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam) Thansk for these great repos!
