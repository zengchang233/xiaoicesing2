import math
from torch.optim.lr_scheduler import _LRScheduler
import torch

class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            optimizer,
            warmup_steps = 25000,
            last_epoch = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [
                lr * step_num ** -0.5
                for lr in self.base_lrs
            ]
        else:
            return [
                lr
                * self.warmup_steps ** 0.5
                * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step

class BaseClass:
    '''
    Base Class for learning rate scheduler
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        '''
        warm_up_epoch: the first warm_up_epoch is the multiprocess warm-up stage
        scale_ratio: multiplied to the current lr in the multiprocess training
        process
        '''
        self.optimizer = optimizer
        self.max_iter = num_epochs * epoch_iter
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.scale_ratio = scale_ratio
        self.current_iter = 0
        self.warm_up_iter = warm_up_epoch * epoch_iter
        self.warm_from_zero = warm_from_zero

    def get_multi_process_coeff(self):
        lr_coeff = 1.0 * self.scale_ratio
        if self.current_iter < self.warm_up_iter:
            if self.warm_from_zero:
                lr_coeff = self.scale_ratio * self.current_iter / self.warm_up_iter
            elif self.scale_ratio > 1:
                lr_coeff = (self.scale_ratio -
                            1) * self.current_iter / self.warm_up_iter + 1.0

        return lr_coeff

    def get_current_lr(self):
        '''
        This function should be implemented in the child class
        '''
        return 0.0

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def step(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        self.set_lr()
        self.current_iter += 1

    def step_return_lr(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        current_lr = self.get_current_lr()
        self.current_iter += 1

        return current_lr

class ExponentialDecrease(BaseClass):

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio, warm_from_zero)

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        current_lr = lr_coeff * self.initial_lr * math.exp(
            (self.current_iter / self.max_iter) *
            math.log(self.final_lr / self.initial_lr))
        return current_lr

class TriAngular2(BaseClass):
    '''
    The implementation of https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 cycle_step=2,
                 reduce_lr_diff_ratio=0.5):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio)

        self.reduce_lr_diff_ratio = reduce_lr_diff_ratio
        self.cycle_iter = cycle_step * epoch_iter
        self.step_size = self.cycle_iter // 2

        self.max_lr = initial_lr
        self.min_lr = final_lr
        self.gap = self.max_lr - self.min_lr

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        point = self.current_iter % self.cycle_iter
        cycle_index = self.current_iter // self.cycle_iter

        self.max_lr = self.min_lr + self.gap * self.reduce_lr_diff_ratio**cycle_index

        if point <= self.step_size:
            current_lr = self.min_lr + (self.max_lr -
                                        self.min_lr) * point / self.step_size
        else:
            current_lr = self.max_lr - (self.max_lr - self.min_lr) * (
                point - self.step_size) / self.step_size

        current_lr = lr_coeff * current_lr

        return current_lr


def show_lr_curve(scheduler):
    import matplotlib.pyplot as plt

    lr_list = []
    for current_lr in range(0, scheduler.max_iter):
        lr_list.append(scheduler.step_return_lr(current_lr))
    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, '-o', markersize=1)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.show()


if __name__ == '__main__':
    optimizer = None
    num_epochs = 6
    epoch_iter = 500
    initial_lr = 0.6
    final_lr = 0.1
    warm_up_epoch = 2
    scale_ratio = 4
    scheduler = ExponentialDecrease(optimizer, num_epochs, epoch_iter,
                                    initial_lr, final_lr, warm_up_epoch,
                                    scale_ratio)
    # scheduler = TriAngular2(optimizer,
    #                         num_epochs,
    #                         epoch_iter,
    #                         initial_lr,
    #                         final_lr,
    #                         warm_up_epoch,
    #                         scale_ratio,
    #                         cycle_step=2,
    #                         reduce_lr_diff_ratio=0.5)

    show_lr_curve(scheduler)
