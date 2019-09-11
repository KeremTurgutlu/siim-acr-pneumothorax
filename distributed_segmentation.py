# Distributed training for classification
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
from sklearn.model_selection import KFold
from fastai.callbacks.hooks import model_sizes
from losses import *
from unet_models import *
from callbacks import *
kfold = KFold(n_splits=5, random_state=42)
torch.backends.cudnn.benchmark = True


class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)

    
class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList

    
def get_data(sz=1024, bs=1, chexpert=True, fold_idx=0):
    data_path = Path("/home/turgutluk/data/siim_acr_pneu/")
    get_y_fn = lambda x: data_path/f'train/masks_{sz}/{Path(x).stem}.png'
    codes = ['void', 'pthorax']
    tfms = get_transforms()
    
    items = (SegmentationItemList.from_csv(data_path, 'seg_df.csv', folder=f'train/images_{sz}', suffix='.jpg'))
    trn_idxs, val_idxs = list(kfold.split(range(len(items))))[fold_idx]
    data = (items.split_by_idxs(trn_idxs, val_idxs)
            .label_from_func(get_y_fn, classes=codes)
            .transform(tfms=tfms, size=sz, tfm_y=True, resize_method=ResizeMethod.NO, padding_mode='reflection')
            .databunch(bs=bs))
    chx_mimic_stats = [tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225])]
    data = data.normalize(chx_mimic_stats) if chexpert else data.normalize(imagenet_stats)
    return data


def _xresnet_split(m:nn.Module): return (m[6], m[8])

def get_learn(data, chexpert=True, pretrained=True):
    # denseunet121
    learn = densenet121_unet_leaner(data, "/home/turgutluk/data/models/chx-mimic-only-pneumo-densenet121-320.pth")
    learn.to_fp16(); learn.freeze() 
    learn.loss_func, learn.metrics = dice_loss, [dice]
    return learn


@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        lr: Param("Learning rate", float)=3e-3,
        sz: Param("Size (px: 128,192,224)", int)=448,
        bs: Param("Batch size", int)=2,
        fold_idx: Param("Fold idx", int)=0,
        pretrained: Param("Use non-xresnet pretrained model", bool)=True,
        chexpert: Param("Use chexpert pretrained model", bool)=True,
        epochs: Param("Number of epochs", int)=15,
        ):
    """
    python /home/turgutluk/fastai/fastai/launch.py --gpus=0123457 distributed_segmentation.py
    sz: 224 bs: 8
    sz: 448 bs: 2
    """

    # Pick one of these
    gpu = setup_distrib(gpu)
    data = get_data(sz, bs, chexpert, fold_idx)
    learn = get_learn(data, chexpert, pretrained)
    learn.to_distributed(gpu)
    learn.callbacks.append(SaveModelCallback(learn, monitor='dice', name=f"seg-densenet121-{sz}-{fold_idx}"))
    
    learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7))
    learn.unfreeze()
    learn.load(f"seg-densenet121-{sz}-{fold_idx}")
    learn.fit_one_cycle(epochs, lr/10, moms=(0.8,0.7))
    
    
    
    
    