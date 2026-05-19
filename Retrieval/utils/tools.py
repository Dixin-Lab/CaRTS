import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
plt.switch_backend('agg')


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.is_pretraining = args.is_pretraining
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.rank = args.rank
        else:
            self.rank = None

    def __call__(self, val_loss, model, path):
        if self.is_pretraining:
            score = -val_loss
        else:
            score = -abs(val_loss-1)
        if self.best_score is None:
            self.best_score = score
            self.best_score = score
            if self.verbose:
                if (self.use_multi_gpu and self.rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
            if self.use_multi_gpu:
                dist.barrier()
                if self.rank == 0:
                    self.save_checkpoint(val_loss, model.module, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.use_multi_gpu:
                dist.barrier()
                if self.rank == 0:
                    self.save_checkpoint(val_loss, model.module, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                if (self.use_multi_gpu and self.rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
