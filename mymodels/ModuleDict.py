import jittor as jt
from jittor import nn

class ModuleDict(jt.Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules_dict = {}
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self.modules_dict[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self.modules_dict[key]
        delattr(self, key)

    def __len__(self):
        return len(self.modules_dict)

    def __iter__(self):
        return iter(self.modules_dict)

    def __contains__(self, key):
        return key in self.modules_dict

    def add_module(self, key, module):
        if not isinstance(module, nn.Module):
            raise TypeError(f"ModuleDict values must be of type nn.Module, but got {type(module)}")
        self.modules_dict[key] = module
        setattr(self, key, module)  

    def keys(self):
        return self.modules_dict.keys()

    def items(self):
        return self.modules_dict.items()

    def values(self):
        return self.modules_dict.values()

    def update(self, modules):
        if isinstance(modules, dict):
            for key, module in modules.items():
                self.add_module(key, module)
        else:
            raise TypeError("ModuleDict update should be called with a dict of key/module pairs.")

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.modules_dict.items()}

    def load_state_dict(self, state_dict):
        for key, state in state_dict.items():
            if key in self.modules_dict:
                self.modules_dict[key].load_state_dict(state)
            

    def train(self):
        for module in self.modules_dict.values():
            module.train()

    def eval(self):
        for module in self.modules_dict.values():
            module.eval()

    @property
    def training(self):
        return all(module.training for module in self.modules_dict.values())
