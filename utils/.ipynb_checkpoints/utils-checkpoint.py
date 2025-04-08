'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import yaml

def load_config(path):
    
    with open(path) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    return config
    
def get_infos(infos, nb=4, clean=True,samples=None):
    # prompt=""
    information = []
    obs = []
    for _i in range(nb):
        dico = {}
        if True:
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd.split()[0] in []]:
                    commands_.remove(cmd_)

        dico["goal"] = infos["goal"][_i]
        dico["obs"] = infos["description"][_i]
        if "context" in infos.keys():
            dico["context"] = infos["context"][_i]
        dico["possible_actions"] = infos["admissible_commands"][_i]
        dico["won"] = infos["won"][_i]
        obs.append(dico["obs"])
        if samples:
            dico["samples"]=samples
        information.append(dico)
    return obs, information
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]