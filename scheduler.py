import jittor as jt

class GradualWarmupScheduler:
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.base_lrs = []
        for param_group in optimizer.param_groups:
            if 'lr' not in param_group:
                param_group['lr'] = 0.0001 
            self.base_lrs.append(param_group['lr'])
        self.last_epoch = 0

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                    
                    for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                        param_group['lr'] = base_lr * self.multiplier
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        if self.last_epoch <= self.total_epoch:
            warmup_lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step()
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        cos_inner = (1 + jt.cos(jt.pi * self.last_epoch / self.T_max)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * cos_inner for base_lr in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr