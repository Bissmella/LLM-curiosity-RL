'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import torch

from . import discount_cumsum, combined_shape
import torch.nn.functional as F

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95, intrinsic_reward=False):
        self.intrinsic_reward= intrinsic_reward    #Use intrinsic reward based on novelty of actions
        self.obs_buf = [None for _ in range(size)]
        self.possible_act_buf = [None for _ in range(size)]
        self.cmd_buf = [None for _ in range(size)]
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        

    def store(self, obs, possible_act, cmd, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.possible_act_buf[self.ptr] = possible_act
        self.cmd_buf[self.ptr] = cmd
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        self.action_counts = {}
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        if self.intrinsic_reward:
            intrinsic_rews = np.array([
                                        self.compute_intrinsic_reward(self.cmd_buf[i])
                                        for i in range(self.path_start_idx, self.ptr)
                                    ])
            intrinsic_rews = np.append(intrinsic_rews, 0)
            # the next two lines implement GAE-Lambda advantage calculation
            rews = rews + intrinsic_rews
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, possible_act=self.possible_act_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }
    
    def compute_intrinsic_reward(self, action, scale=0.0096):
        # Update count
        if action not in self.action_counts:
            self.action_counts[action] = 0
        self.action_counts[action] += 1

        # Return inverse square root reward
        return scale * 1 / np.sqrt(self.action_counts[action])
    

class PPOBufferAugmented:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    It is augmented with goals and trajectory lengths
    """

    def __init__(self, size, gamma=0.99, lam=0.95, intrinsic_reward=False, dualValue=False, intrinsic_decay=False, intrinsic_decay_scale=40):
        self.intrinsic_reward= intrinsic_reward    #Use intrinsic reward based on novelty of actions
        self.dualValue = dualValue
        self.intrinsic_decay = intrinsic_decay
        self.intrinsic_decay_scale = intrinsic_decay_scale
        self.obs_buf = [None for _ in range(size)]
        self.possible_act_buf = [None for _ in range(size)]
        self.goal_buf = []#[None for _ in range(size)]
        self.traj_lens = []#[None for _ in range(size)]
        self.terminals = []#[None for _ in range(size)]
        self.cmd_buf = [None for _ in range(size)]
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        if self.intrinsic_reward and self.dualValue:
            self.valCur_buf = np.full(size, np.nan, dtype=np.float32)#np.zeros(size, dtype=np.float32)
            self.retCur_buf = np.full(size, np.nan, dtype=np.float32)#np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.freq_reward = True  #TODO hardcoded

    def store(self, obs, possible_act, cmd, act, rew, val, logp, val_cur=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.possible_act_buf[self.ptr] = possible_act
        self.cmd_buf[self.ptr] = cmd
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        if val_cur is not None:
            self.valCur_buf[self.ptr] = val_cur
        self.ptr += 1

    def store_goal(self, goal, ep_len, ter=None):
        """
        goal: trajectory goal, string
        ep_len: episode length, int
        """
        self.goal_buf.append(goal)
        self.traj_lens.append(ep_len)
        if ter is not None:
            self.terminals.append(ter)
    def reset_goal(self):
        self.goal_buf=[]
        self.traj_lens=[]
        self.terminals=[]
    def finish_path(self, last_val=0, last_curVal=0, win=False, epoch=0, cur_model=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        self.action_counts = {}
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        if cur_model is None or not self.freq_reward:
            scale = 0.0096
        else:
            scale = 0.006
        if not win:
            if self.dualValue:
                cur_vals = np.append(self.valCur_buf[path_slice], last_curVal)
            if self.intrinsic_reward:
                progress = torch.tensor(epoch / 150.0)
                sharpness = torch.tensor(5.0)
                lambda_int = scale * torch.sigmoid((progress - 0.5) * sharpness).item()
                if self.freq_reward:
                    freq_rews = np.array([
                                                self.compute_intrinsic_reward(self.cmd_buf[i], scale=lambda_int) #TODO hardcoded coefficient
                                                for i in range(self.path_start_idx, self.ptr)
                                            ])
                    freq_rews = np.append(freq_rews, 0)
                # curiosity reward based on temporal predictability
                if cur_model is not None:
                    goal = self.goal_buf[-1]   #goal string
                    actions = self.cmd_buf[path_slice]   #list of actions
                    cur_reward = cur_model.compute_novelty(goal, actions) * lambda_int#TODO hardcoded coefficient
                    tgt_shape = rews.shape[0] - 1
                    cur_reward = np.pad(cur_reward, (0, tgt_shape - cur_reward.shape[0])) if cur_reward.shape[0] < tgt_shape else cur_reward
                    if self.freq_reward:
                        intrinsic_rews = freq_rews + np.append(cur_reward, 0)
                    else:
                        cur_reward = np.append(cur_reward, 0)
                        intrinsic_rews = cur_reward
                    # try:
                    #     intrinsic_rews[:-1] += cur_reward   #cur_reward is one element less
                    # except:
                    #     print("Error occurred! Goal:", goal)
                    #     print("Actions:", actions)
                
                #intrinsic_rews = intrinsic_rews * (40 / (40 + epoch))
                # the next two lines implement GAE-Lambda advantage calculation
                if self.dualValue:
                    intrinsic_deltas = intrinsic_rews[:-1] + self.gamma * cur_vals[1:] - cur_vals[:-1]
                    if self.intrinsic_decay:
                        if epoch < 40:
                            decay_factor = 1
                        else:
                            decay_factor = self.intrinsic_decay_scale/ (self.intrinsic_decay_scale + epoch)
                        intrinsic_deltas = intrinsic_deltas * decay_factor
                    self.retCur_buf[path_slice] = discount_cumsum(intrinsic_rews, self.gamma)[:-1]

                
                #actions
                #rews = rews + intrinsic_rews
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        if not win and self.intrinsic_reward:
            if self.dualValue:
                self.adv_buf[path_slice] += discount_cumsum(intrinsic_deltas, self.gamma * self.lam)
            else:
                try:
                    self.adv_buf[path_slice] += intrinsic_rews[:-1] ##discount_cumsum(intrinsic_rews, self.gamma)[:-1]#
                except:
                    print( "error encountered: ")

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        if self.dualValue:
            data = dict(obs=self.obs_buf, possible_act=self.possible_act_buf, act=self.act_buf, ret=self.ret_buf, retCur_buf=self.retCur_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf, valCur_buf=self.valCur_buf)
        else:
            data = dict(obs=self.obs_buf, possible_act=self.possible_act_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }
    
    def compute_intrinsic_reward(self, action, scale=0.0096):
        # Update count
        if action not in self.action_counts:
            self.action_counts[action] = 0
        self.action_counts[action] += 1

        # Return inverse square root reward
        return scale * 1 / np.sqrt(self.action_counts[action])