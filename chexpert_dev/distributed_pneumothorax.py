# Distributed training for pneumothorax classification with Chexpert + MIMIC
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True


def get_data(sz=320, bs=64, pretrained=True, binary=False):
    
    chexpert_path = Path("/home/turgutluk/data/chexpert/")
    mimic_path = Path("/home/turgutluk/data/mimic-cxr")
    train_df = pd.read_csv(mimic_path/"chx_mimic_train_320.csv")
    valid_df = pd.read_csv(mimic_path/"chx_mimic_valid_320.csv")
    
    cols = ['Path', 'Pneumothorax']
    train_df, valid_df = train_df[cols], valid_df[cols]
    train_valid_df = pd.concat([train_df, valid_df])

    data_path = Path("/home/turgutluk/data")
    tfms = get_transforms()
    
    if binary:
        train_valid_df = train_valid_df[train_valid_df.Pneumothorax != -1].reset_index(drop=True)

    data = (ImageList.from_df(train_valid_df, data_path, cols=['Path'])
            .split_by_rand_pct(valid_pct=0.02, seed=42)
            .label_from_df(label_cls=CategoryList)
            .transform(tfms, size=sz)
            .databunch(bs=bs))
    data = data.normalize(imagenet_stats) if pretrained else data.normalize()
    return data


def get_learn(data, pretrained=True, binary=False):
    if pretrained: learn = cnn_learner(data, models.densenet121, pretrained=pretrained).to_fp16()
    else: learn = Learner(data, models.xresnet34(c_out=data.c)).to_fp16()
    if binary: learn.metrics = [accuracy, AUROC()]
    else: learn.metrics = [accuracy]
    return learn


@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        lr: Param("Learning rate", float)=3e-2,
        sz: Param("Size (px: 128,192,224)", int)=320,
        bs: Param("Batch size", int)=64,
        pretrained: Param("Use ImageNet pretrained model", bool)=True,
        binary: Param("Do binary classification", bool)=True,
        ):
    """
    python /home/turgutluk/fastai/fastai/launch.py --gpus=0123457 distributed_pneumothorax.py
    """

    # Pick one of these
    gpu = setup_distrib(gpu)
    data = get_data(sz, bs, pretrained)
    learn = get_learn(data, pretrained)
    learn.to_distributed(gpu)
    
    if not pretrained:
        learn.fit_one_cycle(10, lr)
        learn.to_fp32().save(f"chx-mimic-only-pneumo-xresnet34-{sz}", with_opt=False)
    else:
        learn.fit_one_cycle(5, lr)
        learn.freeze_to(-1)
        learn.fit_one_cycle(5, slice(lr/5))
        learn.unfreeze()
        learn.fit_one_cycle(5, slice(lr/10))
        learn.to_fp32().save(f"chx-mimic-only-pneumo-densenet121-{sz}", with_opt=False)
    