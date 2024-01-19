#========================================================================
# Classification with different methods
# References:
# Medical MAE: https://github.com/lambert-x/medical_mae
# MGCA: https://github.com/HKU-MedAI/MGCA
# BioViL: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized
# MedKLIP: https://github.com/MediaBrain-SJTU/MedKLIP
# We thanks for all of their great works!

# by Jingfeng Yao, from HUST-VL (https://github.com/hustvl)
#========================================================================



#========================================================================
# CovidX3 (30k for training in total, 0.4k for testing)
# Classification with different data percentage (0.01, 0.1, 1)
# all the results will be saved in ./output/covidx3
# We run each experiments for 3 times and report the average results
#========================================================================
sh train_files/eva_x/covidx3/eva_x.sh           # eva_x (ours)

sh train_files/densnet121/covidx3/mocov2.sh     # reproduce of other methods
sh train_files/densnet121/covidx3/medmae.sh
sh train_files/resnet50/covidx3/biovil.sh
sh train_files/resnet50/covidx3/mgca.sh
sh train_files/resnet50/covidx3/medklip.sh
sh train_files/vit/covidx3/medmae.sh



#========================================================================
# CovidX4 (67k for training in total, 8k for testing)
# Classification with different data percentage (0.01, 0.1, 1)
# all the results will be saved in ./output/covidx4
# We run each experiments for 3 times and report the average results
#========================================================================
sh train_files/eva_x/covidx4/eva_x.sh           # eva_x (ours)

sh train_files/densnet121/covidx4/mocov2.sh     # reproduce of other methods
sh train_files/densnet121/covidx4/medmae.sh
sh train_files/resnet50/covidx4/biovil.sh
sh train_files/resnet50/covidx4/mgca.sh
sh train_files/resnet50/covidx4/medklip.sh
sh train_files/vit/covidx4/medmae.sh