# siim-acr-pneumothorax
Kaggle competition repo: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/

- CXU: Chexnet U-ignore Pretraining
- R34: ResNet34
- RX34: XResNet34
- DN121: DenseNet121
- POST: Pixel Removal Post Processing
- GS: Grid Search for optimal seg and clas thres

# Experiments

|   | Local       | Public LB | Method   | Input Size (CLS - SEG) | 
|---|-------------|------------|-----------| -----------|
|   | - |   0.7886    | All -1 |
|   |   0.8094          |    0.8108        |  CXU R34 CLS + CXU RX34 SS (GS - CLS ONLY) | 224-224   |
|   |    0.8017         |   0.8110         | CXU RX34 CLS + CXU RX34 SS (GS - CLS ONLY) | 224-224        |
|   |    0.8090         |   0.8130         | TTA_cls - (CXU R34 RX34) - GS | 224-224 |
|   |    0.8150         |   0.8168         | TTA_cls - POST - (CXU R34 RX34) - GS | 224-224 |
|   |    0.8091         |   0.8169         | TTA_cls - POST - (CXU RX34 RX34) - GS | 224-224 |
|   |    0.8117         |   0.8202         | TTA_cls_seg - POST - (CXU RX34 RX34) - GS | 224-224 |
|   |    0.8201         |   0.8161         | TTA_cls_seg - POST - (CXU R34 RX34) - GS | 224-224 |
|   |    0.8190         |   0.8096         | TTA_cls_seg - POST - (DN121 RX34) - GS | 224-224 |
|   |    0.8232         |   0.8186         | Average Ensemble Before GS| 224-224 |
