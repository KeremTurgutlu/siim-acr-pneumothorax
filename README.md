# siim-acr-pneumothorax
Kaggle competition repo: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/

- CX: Chexnet
- R34: ResNet34
- RX34: XResNet34
- SS: Semantic Segmentation
- IS: Instance Segmentation

# Experiments

|   | Local       | Public LB | Method   | 
|---|-------------|------------|-----------|
|   | - |   0.7886    | All -1 |
|   |   0.8094          |    0.8108        |  CX R34 CLS + CX RX34 SS (GS - CLS ONLY)      |
|   |    0.8017         |   0.8110         | CX RX34 CLS + CX RX34 SS (GS - CLS ONLY)         |
