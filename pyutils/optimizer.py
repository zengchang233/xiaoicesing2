import torch
import numpy as np
from ipdb import set_trace
from torch.optim import (
    SGD,
    Adam,
    AdamW,
    RMSprop,
    RAdam,
    NAdam,
    ASGD
)

class NoamOpt():
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, model_size, warmup, factor = 1.0):
        '''
        model_size: d_model
        factor: factor
        warmup: warmup step
        optimizer: optimizer (Adam default)
        '''
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)

class ScheduledOptimD():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, init_lr, n_warmup_steps, current_steps):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = init_lr

    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self.optimizer.step()

    def step_and_update_lr(self):
        self._update_learning_rate()
        self.optimizer.step()

    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def zero_grad(self):
        # print(self.init_lr)
        self.optimizer.zero_grad()

    def set_current_steps(self, step):
        self.n_current_steps = step

    def _get_lr_scale(self):
        #  set_trace()
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            "_step": self.n_current_steps,
            "warmup": self.n_warmup_steps,
            "factor": self.init_lr,
            "_rate": self.get_learning_rate(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)

def get_g_opt(model, optim, d_model, warmup, factor):
    base = torch.optim.Adam(model.parameters(), lr = 0, betas = (0.9, 0.98), eps = 1e-9)
    return NoamOpt(base, d_model, warmup, factor)

def get_d_opt(model, optim, warmup, factor, current_step):
    base = torch.optim.Adam(model.parameters(), lr = 0, betas = (0.9, 0.98), eps = 1e-9)
    return ScheduledOptimD(base, factor, warmup, current_step)
