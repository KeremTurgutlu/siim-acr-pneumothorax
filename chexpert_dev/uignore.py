from fastai.vision import *
from sklearn.metrics import roc_auc_score


__all__ = ['MaskedBCEWithLogitsLoss', 'MaskedMultiLabelROCAUC', 'masked_accuracy_thres']


class MaskedBCEWithLogitsLoss(Module):
    "mask with labels == m"
    def __init__(self, m=-1):
        self.m = m
        self.eps = 1e-8
        
    def forward(self, input, target):

        l = (target*torch.log(torch.sigmoid(input).clamp(self.eps, 1-self.eps)) +
                    (1-target)*torch.log((1 - torch.sigmoid(input)).clamp(self.eps, 1-self.eps)))
        return -((target != self.m).float()*l).sum(1).mean()

    
class MaskedMultiLabelROCAUC(LearnerCallback):
    "Computes the ROC AUC for multilabel classification"
    _order = -20 
    def __init__(self, learn, eps=1e-15, sigmoid=True, verbose=False, m=-1):
        "ignore target values equal to m"
        super().__init__(learn)
        self.eps, self.sigmoid, self.verbose = eps, sigmoid, verbose
        self.c = self.learn.data.c
        self.m = m

    def on_train_begin(self, **kwargs):
        if not self.verbose: 
            self.learn.recorder.add_metric_names([f'mean_roc_auc'])
        else: 
            self.learn.recorder.add_metric_names([f"roc_auc_{c}" for c in self.learn.data.classes] +
                                                   ['mean_roc_auc'])

    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = LongTensor([]), Tensor([])
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.sigmoid: last_output = torch.sigmoid(last_output)
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu().long()))
    
    def on_epoch_end(self, last_metrics, **kwargs):        
        roc_auc = []
        for i in range(self.c):
            mask = self.targs[:,i] != self.m
            targs = self.targs[:,i][mask]
            preds = self.preds[:,i][mask]
            if torch.equal(targs.unique(), tensor([0,1])):
                roc_auc.append(roc_auc_score(targs, preds))
            else:
                roc_auc.append(tensor(np.nan))
        
        roc_auc = [tensor(m) for m in roc_auc]
        t = tensor(roc_auc)
        
        mean_roc_auc = t[~torch.isnan(t)].mean()
        if not self.verbose: res = mean_roc_auc
        else: res = roc_auc + [mean_roc_auc]
        return add_metrics(last_metrics, res)
    
    
def masked_accuracy_thres(input, target, thres=0.5, m=-1):
    mask = target != m
    preds = (input.sigmoid() > thres)[mask].float()
    targs = target[mask].cuda()
    return (preds == targs).float().mean()