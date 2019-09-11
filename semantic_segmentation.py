from fastai.vision import *
from fastai.vision.interpret import *
import fastai
from fastai.script import *
from fastai.distributed import *


class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)

class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList



@call_parse
def main(gpu:Param("GPU to run on", str)=None,
         fold_idx:Param("Kfold idx", int)=0):
    
    """
    python ../../fastai/fastai/launch.py --gpus=2457 ./semantic_segmentation.py
    """
    
    # Init
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    bs,sz = 2,1024
    # fold_idx = 4 # for kfold

    # ### databunch
    data_path = Path("../../data/siim_acr_pneu/"); data_path.ls()
    chexpert_stats = [tensor([0.5381, 0.5381, 0.5381]), tensor([0.2820, 0.2820, 0.2820])]

    tfms = get_transforms()


    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, random_state=42)

    get_y_fn = lambda x: data_path/f'train/masks_{sz}/{Path(x).stem}.png'
    codes = ['void', 'pthorax']

    items = (SegmentationItemList.from_csv(data_path, 'seg_df.csv', folder=f'train/images_{sz}', suffix='.jpg'))
    trn_idxs, val_idxs = list(kfold.split(range(len(items))))[fold_idx]
    data = (items.split_by_idxs(trn_idxs, val_idxs)
            .label_from_func(get_y_fn, classes=codes)
            .transform(tfms=tfms, size=sz, tfm_y=True, resize_method=ResizeMethod.NO, padding_mode='reflection')
            .databunch(bs=bs)
            .normalize(chexpert_stats))


    def xresnet50(pretrained=True):
        return models.xresnet50(pretrained=pretrained)

    def xresnet34(pretrained=True):
        return models.xresnet34(pretrained=pretrained)


    # INIT LEARN
    # learn = unet_learner(data, models.resnet34, pretrained=False, metrics=dice)
    learn = unet_learner(data, xresnet34, pretrained=False, metrics=dice, self_attention=False)


    # LOAD PRETRAINED WEIGHTS
    # chexpert_state_dict = torch.load(f"../../data/chexpert/models/u-ignore-resnet34-{sz}.pth")
    # chexpert_state_dict = torch.load(f"../../data/chexpert/models/u-ignore-resnext34-{sz}.pth")
    clas_state_dict = torch.load(f"../../data/siim_acr_pneu/models/clas-chexpert-ft-resnext34-{sz}.pth")

    # load state dict from chexpert model until last fully connected layer
    for n, p in list(learn.model[0].named_parameters()):
    #     p.data = chexpert_state_dict['model']['0.'+n].to(torch.device('cuda'))
    #     p.data = chexpert_state_dict['model'][n].to(torch.device('cuda'))
        p.data = clas_state_dict['model'][n].data.to('cuda')
    # empty cache to use free gpu
#     del clas_state_dict; gc.collect(); torch.cuda.empty_cache() # doesn't free memory!?

    learn.load(f"seg-chexpert-ft-resnext34-448-{fold_idx}");    

    from callbacks import SaveModelCallback
    learn.callbacks.append(SaveModelCallback(learn, monitor='dice', 
                                name=f"seg-chexpert-ft-resnext34-{sz}-{fold_idx}"))
    learn.train_bn = False
    learn.to_fp16(); learn.freeze()
    from losses import dice_loss
    learn.loss_func = dice_loss
    learn.to_distributed(gpu)


    lr = 1e-4
    learn.fit_one_cycle(10, lr, moms=(0.8,0.7))
    learn.unfreeze()
    lr /= 2
    learn.fit_one_cycle(10, slice(lr), moms=(0.8,0.7))

    if gpu == 2:
        # ADD TEST
        learn.to_fp32();
        test = ImageList.from_folder(data_path/f'test/images_{sz}', extensions='.jpg')
        learn.data.add_test(test, tfm_y=False)
        learn.load(f"seg-chexpert-ft-resnext34-{sz}-{fold_idx}");


        # SAVE LEARN OBJ
        os.makedirs(data_path/"learn", exist_ok=True)
        with ModelOnCPU(learn.model) as model:
            try_save({"data":learn.data, "model":model}, data_path,
                        f"learn/seg-chexpert-ft-resnext34-{sz}-{fold_idx}")
