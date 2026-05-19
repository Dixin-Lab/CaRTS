import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.distributed as dist
plt.switch_backend('agg')


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.args = args
        self.patience = self.args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.args.use_multi_gpu:
                dist.barrier()
            if (self.args.use_multi_gpu and self.args.rank == 0):
                self.save_checkpoint(val_loss, model.module, path)
            elif not self.args.use_multi_gpu:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.args.use_multi_gpu and self.args.rank == 0) or not self.args.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.args.use_multi_gpu:
                dist.barrier()
            if (self.args.use_multi_gpu and self.args.rank == 0):
                self.save_checkpoint(val_loss, model.module, path)
            elif not self.args.use_multi_gpu:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')