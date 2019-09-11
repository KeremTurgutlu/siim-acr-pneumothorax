# Distributed training for classification
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True


def get_data(sz=1024, bs=4, chexpert=True):
    data_path = Path("/home/turgutluk/data/siim_acr_pneu/")
    tfms = get_transforms()
    data = (ImageList.from_csv(data_path, 'clas_df.csv', folder=f'train/images_{sz}', suffix='.jpg')
            .split_by_rand_pct(0.1, seed=42)
            .label_from_df()
            .transform(tfms=tfms, size=sz) 
            .databunch(bs=bs)
     )
    
    chx_mimic_stats = [tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225])]
    data = data.normalize(chx_mimic_stats) if chexpert else data.normalize(imagenet_stats)
    return data


def _xresnet_split(m:nn.Module):
    return (m[6], m[8])


def get_learn(data, chexpert=True, pretrained=True):
    if pretrained: learn = cnn_learner(data, models.densenet121, pretrained=pretrained)
    else: learn = Learner(data, models.xresnet34(c_out=data.c)); learn.split(_xresnet_split)
    learn.metrics = [accuracy, AUROC()]
    # load state dict
    if chexpert:
        if pretrained:
            chx_state_dict = torch.load("/home/turgutluk/data/models/chx-mimic-only-pneumo-densenet121-320.pth")
        else: pass
        for n, p in list(learn.model[:-1].named_parameters()): p.data = chx_state_dict[n].to(torch.device('cuda'))
        
    learn.to_fp16(); learn.freeze()
    learn.metrics = [accuracy, AUROC()]
    return learn


@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        lr: Param("Learning rate", float)=3e-3,
        sz: Param("Size (px: 128,192,224)", int)=1024,
        bs: Param("Batch size", int)=4,
        pretrained: Param("Use non-xresnet pretrained model", bool)=True,
        chexpert: Param("Use chexpert pretrained model", bool)=True,
        ):
    """
    python /home/turgutluk/fastai/fastai/launch.py --gpus=0123457 distributed_classification.py
        sz:448, bs:32 ETA: 0:30 / epoch 320 -> 448 auc: 0.9240
        sz:1024, bs:4 ETA: 4:00 / epoch 320 -> 1024 auc: 0.9290
    """

    # Pick one of these
    gpu = setup_distrib(gpu)
    data = get_data(sz, bs, chexpert)
    learn = get_learn(data, chexpert, pretrained)
    learn.to_distributed(gpu)

    learn.fit_one_cycle(5, lr)
    learn.freeze_to(-1)
    learn.fit_one_cycle(5, slice(lr/5))
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(lr/10))
    learn.to_fp32().save(f"clas-densenet121-{sz}")
    