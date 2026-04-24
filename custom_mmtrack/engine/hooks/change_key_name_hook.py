

from mmtrack.registry import HOOKS
from mmengine.hooks import Hook

import os
import re
from collections import OrderedDict
import torch



@HOOKS.register_module()
class ChangeKeyNameHook(Hook):
    def __init__(self, model_path=None):
        #self.model_path = model_path

        self.revise_keys = [(r'blocks.\d+.', r'\g<0>TransformerEncoderLayer.'),
                       (r'TransformerEncoderLayer.TransformerEncoderLayer.', r'TransformerEncoderLayer.'),
                       ('attn.qkv.weight', 'self_attn.in_proj_weight'),
                       ('attn.qkv.bias', 'self_attn.in_proj_bias'),
                       ('attn.proj', 'self_attn.out_proj'),
                       ('mlp.fc1', 'linear1'),
                       ('mlp.fc2', 'linear2')
                        ]
        self.update_model_checkpoint(model_path)
        #def before_run(self, runner):
    def update_model_checkpoint(self, ckpt_path):

        target_file = ckpt_path.replace('.pth', '_evo.pth')
        #if os.path.exists(target_file) or not os.path.exists(ckpt_path):
        #    return None

        full_state_dict = torch.load(ckpt_path)
        input_state_dict = full_state_dict['state_dict']
        output_state_dict = OrderedDict()

        for old_key, value in input_state_dict.items():

            for pattern, replacement in self.revise_keys:
                old_key = re.sub(pattern, replacement, old_key)
            output_state_dict[old_key] = value

        full_state_dict['state_dict'] = output_state_dict
        torch.save(full_state_dict, target_file)



