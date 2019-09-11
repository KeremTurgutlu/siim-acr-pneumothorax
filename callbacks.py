from fastai.callbacks import *

__all__ = ["SaveModelCallback"]

def _on_epoch_end(self, epoch:int, **kwargs:Any)->None:
    "Compare the value monitored to its best score and maybe save the model."
    if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
    else: #every="improvement"
        current = self.get_monitor_value()
        if current is not None and self.operator(current, self.best):
            self.best = current
            self.learn.save(f'{self.name}')
def _on_train_end(self, **kwargs): pass
    
SaveModelCallback.on_epoch_end = _on_epoch_end
SaveModelCallback.on_train_end = _on_train_end
