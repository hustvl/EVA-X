<h1 align="center">EVA-X: Pre-training with Masked Image Modeling</h1>

Here, we provide codes for EVA-X pre-training. We pre-train <img src="figs/logo.png" height="17"> EVA-X representations series with 8/16 A100 GPUs. 

Typically, the use of <img src="figs/logo.png" height="17">EVA-Xs is much cheaper (1/4 RTX-3090 GPUs). If you want to use EVA-X representation for your tasks, we recommand to refer [X-ray classification](../classification/) and [X-ray segmentation]().

## Intallation

1. create conda environment
    ```
    conda create -n evax python=3.9
    conda activate evax
    ```
2. pip install all requirements
   ```
   pip install torch==2.0.1
   pip install -r requirements.txt
   ```
3. install [xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers), [Apex](https://github.com/NVIDIA/apex?tab=readme-ov-file#from-source) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## Tokenizer Preparing

EVA-X is trained with Medical CLIP as a tokenizer of X-rays. Here, we use the [MGCA](https://github.com/HKU-MedAI/MGCA). Download it from its offical repo. **Thanks for their great work!**

Download it [here](https://drive.google.com/drive/folders/15_mP9Lqq2H15R53qlKn3l_xzGVzi9jX9) and put it in ``./tokenizer``

## Data Preparing

EVA-X is pre-trained with three public datasets: [Chest X-Ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data), [CheXpert](https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small) and [MIMIC-CXR-2.0](). 

1. Download these three datasets.

2. Pre-process them by run
    ```
    TODO
    ```
    Test images and Lateral View images are excluded while pre-training.

## Pre-training

Pre-train EVA-X with the following codes, we train them with 8/16 A100 GPUs:
```
TODO
```

| EVA-X Series | Architecture | #Params | Checkpoint | MIM epochs |
|:------------:|:------------:|:-------:|:----------:|:----------:|
| <img src="figs/logo.png" width="18"> EVA-X-Ti    |  ViT-Ti/16   | 6M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt)| 900 |
| <img src="figs/logo.png" width="18">  EVA-X-S    |  ViT-S/16   | 22M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt)| 600 | 
| <img src="figs/logo.png" width="18">  EVA-X-B    |  ViT-B/16   | 86M      |  [🤗download](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_base_patch16_merged520k_mim.pt)| 600 |


## Acknowledgement

The pre-training codes are built upon [EVA-02](https://github.com/baaivision/EVA/tree/master) and [MGCA](https://github.com/HKU-MedAI/MGCA).