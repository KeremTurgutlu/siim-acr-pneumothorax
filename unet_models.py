from fastai.vision import *
from fastai.callbacks.hooks import model_sizes

__all__ = ['densenet121_unet_leaner']
 
def _new_densenet_split(m:nn.Module): return (m[1])

def densenet121_unet_leaner(data, model_pth="/home/turgutluk/data/models/chx-mimic-only-pneumo-densenet121-320.pth"):
    body = create_body(models.densenet121)
    state_dict = torch.load(model_pth)
    for n, p in list(body.named_parameters()): p.data = state_dict['0.' + n]
    body = body[0].cpu()    
    densenet_children = list(body.children())
    unet_densenet_body = nn.Sequential(nn.Sequential(*densenet_children[:3]),
                                   nn.Sequential(*densenet_children[3:5]),
                                   nn.Sequential(*densenet_children[5:7]),
                                   nn.Sequential(*densenet_children[7:9]),
                                   nn.Sequential(*densenet_children[9:]))
    try:size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(models.unet.DynamicUnet(unet_densenet_body, n_classes=data.c, img_size=size), data.device)
    learn = Learner(data, model)
    learn.split(_new_densenet_split)
    apply_init(learn.model[1:], nn.init.kaiming_normal_)
    return learn