import torch
import numpy as np
import torch.nn as nn


class ModuleInfo():
    """
        Save the informations from a module
    """
    def __init__(self, module, name):
        self.type = type(module)
        self.module = module 
        self.info = {'h':0, 'w':0, 'k':0, 'cin':0, 'cout':0, 'input_dim':0}
        self.name = name

    def __str__(self):
        return f"{self.name} | {self.type} | {self.info}"


class ModulesInfo:
    """
    A wrapper for a list of modules.
    Allows to collect data for all modules in one feed-forward.
    Args:
        - model: the model to collect data from
        - modulesInfo: a list of ModuleInfo objects
        - input_img_size: the size of the input image, for the feed-forwarding
    """
    def __init__(self, model, modulesInfo, input_img_size=None):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.modulesInfo = modulesInfo
        if input_img_size:
            input_image = torch.randn(1, 3, input_img_size, input_img_size)#.cuda()
            self.feed(input_image)

    def _make_feed_hook(self, i):
        def hook(m, x, z):
            self.modulesInfo[i].info['input_dim'] = x[0].size()
            self.modulesInfo[i].info['cin'] = int(x[0].size(1))
            self.modulesInfo[i].info['cout'] = int(z.size(1))
            if isinstance(m, nn.Conv2d):
                self.modulesInfo[i].info['h'] = int(x[0].size(2)) if len(x[0].size())>2 else 1
                self.modulesInfo[i].info['w'] = int(x[0].size(3)) if len(x[0].size())>2 else 1
                self.modulesInfo[i].info['k'] = int(m.weight.size(2))
            elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.modulesInfo[i].info['h'] = int(z.size(2)) if len(z.size())>2 else 1
                self.modulesInfo[i].info['w'] = int(z.size(3)) if len(z.size())>2 else 1
                self.modulesInfo[i].info['k'] = int(x[0].size(2))//int(z.size(2))
            else:
                self.modulesInfo[i].info['h'] = int(x[0].size(2)) if len(x[0].size())>2 else 1
                self.modulesInfo[i].info['w'] = int(x[0].size(3)) if len(x[0].size())>2 else 1
        return hook

    def feed(self, input_image):
        hook_handles = [e.module.register_forward_hook(self._make_feed_hook(i)) for i, e in enumerate(self.modulesInfo)]

        if self.device is not None:
            self.model(input_image.to(self.device))
        else:
            self.model(input_image)

        for handle in hook_handles:
            handle.remove()

    def get_info(self, i):
            return self.modulesInfo[i].info
    
    def modules(self):
        return [m for m in self.modulesInfo]

    def __getitem__(self, key):
        if isinstance(key, int): return self.modulesInfo[key]
        elif isinstance(key, str): return [m for m in self.modulesInfo if m.name == key][0]