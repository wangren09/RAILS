import torch
import torch.nn as nn


class PGD:
    def __init__(self, eps=60 / 255., step=20 / 255., max_iter=20, random_init=False,
                 targeted=False, loss_fn=nn.CrossEntropyLoss(), batch_size=64):
        self.eps = eps
        self.step = step
        self.max_iter = max_iter
        self.random_init = random_init
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def attack(self, model, x, y, x_adv=None, targets=None):
        if x_adv is None:
            if self.random_init:
                x_adv = self.step * torch.randn_like(x) + x
                x_adv = x_adv.clamp(0.0, 1.0)
            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        pred_adv = model(x_adv)
        if isinstance(pred_adv,list):
            pred_adv = pred_adv[-1]
        if self.targeted:
            assert targets is not None, "Target labels not found!"
            loss = self.loss_fn(pred_adv, targets)
        else:
            loss = self.loss_fn(pred_adv, y)
        loss.backward()
        pert = self.step * x_adv.grad.sign()
        x_adv = (x_adv + pert).clamp(0.0, 1.0).detach()
        pert = (x_adv - x).clamp(-self.eps, self.eps)
        return x + pert

    def generate(self, model, x, y=None, targets=None, device=torch.device("cpu")):
        model.to(device)
        model.eval()
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i+self.batch_size].to(device)
            if y is None:
                y_batch = model(x_batch)
                if isinstance(y_batch,tuple):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(device)
            else:
                y_batch = y[i: i+self.batch_size].to(device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(model, x_batch, y_batch, targets=targets)
                else:
                    x_adv_batch = self.attack(model, x_batch, y_batch, x_adv_batch, targets=targets)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu()
