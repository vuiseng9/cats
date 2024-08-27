import json

import torch
import torch.nn as nn

import flash_gemv
scap_fc = flash_gemv.gather_transposed_gemv_flag_3d


class CATS_SwiGLU(nn.Module):
    def __init__(self, original, quiet=True):
        super(CATS_SwiGLU, self).__init__()
        self.quiet = quiet
        self.threshold = -1
        self.act_fn = original.act_fn
        self.Wup = original.up_proj.weight.contiguous().data
        self.Wgatet = original.gate_proj.weight.t().contiguous().data
        self.Wdownt = original.down_proj.weight.t().contiguous().data

    def extra_repr(self) -> str:
        return f"threshold: {self.threshold:.5f}"
    
    def forward(self, x):
        assert x.dim() == 3, "assume 3d x"
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
            return self.decode_forward(x)
        return self.prefill_forward(x)
    
    def prefill_forward(self, x):
        z = torch.matmul(x, self.Wup.t()) * self.act_fn(torch.matmul(x, self.Wgatet))
        return torch.matmul(z, self.Wdownt)
    
    def decode_forward(self, x):
        x_1 = self.act_fn(torch.matmul(x, self.Wgatet))
        flags = torch.abs(x_1) > self.threshold
        z = flash_gemv.gather_gemv_elemul_flag_3d(x, x_1, self.Wup, flags)
        return flash_gemv.gather_transposed_gemv_flag_3d(z, self.Wdownt, flags)


class SCAP_SwiGLU(nn.Module):
    def __init__(self, original, quiet=True):
        super(SCAP_SwiGLU, self).__init__()
        self.quiet = quiet
        self.tau_upgate = -1
        self.tau_down = -1
        self.act_fn = original.act_fn
        self.Wupt = original.up_proj.weight.t().contiguous().data
        self.Wgatet = original.gate_proj.weight.t().contiguous().data
        self.Wdownt = original.down_proj.weight.t().contiguous().data

    def extra_repr(self) -> str:
        return f"tau_upgate: {self.tau_upgate:.5f}\ntau_down:   {self.tau_down:.5f}"
    
    def forward(self, x):
        assert x.dim() == 3, "assume 3d x"
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
            return self.decode_forward(x)
        return self.prefill_forward(x)
    
    def prefill_forward(self, x):
        z = torch.matmul(x, self.Wupt) * self.act_fn(torch.matmul(x, self.Wgatet))
        return torch.matmul(z, self.Wdownt)

    def decode_forward(self, x):
        # for debug
        # z = torch.matmul(x, self.Wupt) * self.act_fn(torch.matmul(x, self.Wgatet))
        # y = torch.matmul(z, self.Wdownt)
        
        # unequal_ids = torch.nonzero(y != o, as_tuple=True)
        # y[unequal_ids]

        x_mask = torch.abs(x) > self.tau_upgate      
        gated = scap_fc(x, self.Wupt, x_mask) * self.act_fn(scap_fc(x, self.Wgatet, x_mask))

        gated_mask = torch.abs(gated) > self.tau_down
        return scap_fc(gated, self.Wdownt, gated_mask)
    

def replace_module_with_custom(model, mod_cls, custom_cls, quiet=True):
    for name, module in model.named_children():
        if isinstance(module, mod_cls):
            wrapped_module = custom_cls(module, quiet=quiet)
            setattr(model, name, wrapped_module)
        else:
            replace_module_with_custom(module, mod_cls, custom_cls, quiet=quiet)
    return model


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def set_sparse_swiglu_threshold(model, prune_cfg_pth):
    cfg = read_json_file(prune_cfg_pth)

    if not isinstance(cfg, dict):
        raise TypeError("cfg is not a dict")

    for n, m in model.named_modules():
        if isinstance(m, CATS_SwiGLU):
            k = f'{n}.act_fn'
            m.threshold = cfg['post'][k]['threshold']

        elif isinstance(m, SCAP_SwiGLU):
            upgate_k = f'{n}.up_proj' # up_proj/gate_proj are having same value
            down_k = f'{n}.down_proj'
            m.tau_upgate = cfg['pre'][upgate_k]['threshold']
            m.tau_down = cfg['pre'][down_k]['threshold']

    return None