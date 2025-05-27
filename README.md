# Responsible-AI - Skin Lesion Classification Modekl

This project implements a deep learning pipeline for classifying skin lesions using the HAM10000 dataset. A modified ResNet18 model is used, incorporating both image data and auxiliary patient metadata (age, sex, localization).

## Dataset

We use the **HAM10000 ("Human Against Machine") dataset**, which includes dermatoscopic images of pigmented lesions and corresponding metadata.

**Dataset Info & Source:** 
info: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

**Labels:**
'NMSC': "Non-melanoma skin cancer" (incoludes bcc, akiec, bkl, df, vasc) (label 0)
'mel': "Melanoma" (label 1)
'nv': "Melanocytic nevi" (label 2)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PIL
- pytorch-grad-cam


