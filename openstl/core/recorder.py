# import numpy as np
# import torch
#
#
# class Recorder:
#     def __init__(self, verbose=False, delta=0, early_stop_time=10):
#         self.verbose = verbose
#         self.best_score = None
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.decrease_time = 0
#         self.early_stop_time = early_stop_time
#
#     def __call__(self, val_loss, model, path, early_stop=False):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score >= self.best_score + self.delta:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.decrease_time = 0
#         else:
#             self.decrease_time += 1
#         # return self.decrease_time <= self.early_stop_time if early_stop else 0
#         return True if early_stop else 0
#
#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
#         self.val_loss_min = val_loss

# import numpy as np
# import torch
#
#
# class Recorder:
#     def __init__(self, verbose=False, delta=0, early_stop_time=10):
#         self.verbose = verbose
#         self.best_score = None
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.decrease_time = 0
#         self.early_stop_time = early_stop_time
#
#     def __call__(self, val_loss, model, path, early_stop=False):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score >= self.best_score + self.delta:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.decrease_time = 0
#         else:
#             self.decrease_time += 1
#         # return self.decrease_time <= self.early_stop_time if early_stop else 0
#         return True if early_stop else 0
#
#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
#         self.val_loss_min = val_loss

# import numpy as np
# import torch
#
#
# class Recorder:
#     def __init__(self, verbose=False, delta=0, early_stop_time=10):
#         self.verbose = verbose
#         self.best_score = None
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.decrease_time = 0
#         self.early_stop_time = early_stop_time
#
#     def __call__(self, val_loss, model, path, early_stop=False):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score >= self.best_score + self.delta:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.decrease_time = 0
#         else:
#             self.decrease_time += 1
#         # return self.decrease_time <= self.early_stop_time if early_stop else 0
#         return True if early_stop else 0
#
#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
#         self.val_loss_min = val_loss

import numpy as np
import torch


class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=10):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = early_stop_time
        self.best_epoch = 0  # 记录最佳epoch

    def __call__(self, val_loss, model, path, epoch, early_stop=False):  # 修正：添加epoch参数
        score = -val_loss

        if self.best_score is None:
            # 第一次记录
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.decrease_time = 0
        elif score >= self.best_score + self.delta:
            # 性能提升
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.decrease_time = 0
        else:
            # 性能未提升
            self.decrease_time += 1

        # 早停判断：只有启用早停且连续下降次数达到阈值
        if early_stop and self.decrease_time >= self.early_stop_time:
            if self.verbose:
                print(f'Early stopping triggered: no improvement for {self.early_stop_time} epochs')
                print(f'Best model at epoch {self.best_epoch + 1} with val_loss {self.val_loss_min:.6f}')
            return True
        return False

    def save_checkpoint(self, val_loss, model, path, epoch):
        if self.verbose:
            print(f'Epoch {epoch + 1}: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path + '/checkpoint.pth')
        self.val_loss_min = val_loss
        self.best_epoch = epoch  # 记录最佳epoch
