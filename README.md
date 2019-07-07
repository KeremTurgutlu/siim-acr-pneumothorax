# siim-acr-pneumothorax
Kaggle competition repo: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/

- CX: Chexnet Pretraining
- R34: ResNet34
- RX34: XResNet34


# Experiments

|   | Local       | Public LB | Method   | 
|---|-------------|------------|-----------|
|   | - |   0.7886    | All -1 |
|   |   0.8094          |    0.8108        |  CX R34 CLS + CX RX34 SS (GS - CLS ONLY)      |
|   |    0.8017         |   0.8110         | CX RX34 CLS + CX RX34 SS (GS - CLS ONLY)         |
|   |    0.8090         |   0.8130         | TTA_cls - (CX R34 RX34) - GS |
|   |    0.8150         |   0.8168         | TTA_cls - POST - (CX R34 RX34) - GS |
|   |    0.8091         |   0.8169         | TTA_cls - POST - (CX RX34 RX34) - GS |
|   |    0.8117         |   0.8202         | TTA_cls_seg - POST - (CX RX34 RX34) - GS |
|   |    0.8201         |   0.8161         | TTA_cls_seg - POST - (CX R34 RX34) - GS |
