"""
PPO implementation taken from https://github.com/openai/spinningup
"""

import sys
import os

# Add the root directory (one level up from current file)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from collections import OrderedDict
from typing import List
from torch.nn.functional import log_softmax
import hydra
import cv2 as cv
from PIL import Image
from utils import *
import torch
import bitsandbytes
import numpy as np
import logging
import re
import cv2 
from transformers import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import glob
from tqdm import tqdm
import time
import pickle
import math
import os
import functools as f
from operator import add
import gc
import gym
import textworld.gym
from gym.envs.registration import register, spec, registry
import time
from textworld import EnvInfos
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer
from random import sample
#import wandb
from accelerate import Accelerator
import warnings
from pyvirtualdisplay import Display

import os
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic


import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml

from object_extractor import ObjectExtractor

import torch.distributed as dist

#TODO   SENSITIVE HERE!!!!!!11
from huggingface_hub import login
login("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")


class PPOUpdater(BaseUpdater):
    def __init__(
        self,
        model_type,
        minibatch_size,
        gradient_batch_size,
        quantized_optimizer,
        gradient_minibatch_size=None,
    ):
        super(PPOUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size
        self._quantized_optimizer = quantized_optimizer

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "optimizer"):
            if kwargs["use_all_params_for_optim"]:
                self._iterator_named_trainable_params = (
                    self._llm_module.named_parameters
                )
            else:
                self._iterator_named_trainable_params = (
                    lambda: self._get_trainable_params(self._llm_module, True)
                )

            self._iterator_trainable_params = (
                p for n, p in self._iterator_named_trainable_params()
            )
            if self._quantized_optimizer:
                self.optimizer = bitsandbytes.optim.PagedAdamW8bit(
                    self._iterator_trainable_params, lr=kwargs["lr"]
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self._iterator_trainable_params, lr=kwargs["lr"]
                )

            if os.path.exists(kwargs["loading_path"] + "/optimizer.checkpoint"):
                self.optimizer.load_state_dict(
                    torch.load(kwargs["loading_path"] + "/optimizer.checkpoint")
                )

        current_process_buffer = {}
        for k in ["actions", "advantages", "returns", "logprobs", "values"]:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        epochs_losses = {"value": [], "policy": [], "loss": []}

        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            for step in tqdm(range(n_minibatches)):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size, len(contexts)
                )

                self.optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    (_minibatch_end_idx - _minibatch_start_idx)
                    / self._gradient_batch_size
                )
                for accumulated_batch in tqdm(range(gradient_accumulation_steps)):
                    _start_idx = (
                        _minibatch_start_idx
                        + accumulated_batch * self._gradient_batch_size
                    )
                    _stop_idx = _minibatch_start_idx + min(
                        (accumulated_batch + 1) * self._gradient_batch_size,
                        _minibatch_end_idx,
                    )

                    _contexts = contexts[_start_idx:_stop_idx]
                    _candidates = candidates[_start_idx:_stop_idx]
                    if len(_contexts) == 0:
                        break
                    if self._gradient_minibatch_size is None:
                        _batch_size = sum(len(_c) for _c in _candidates)
                    else:
                        _batch_size = self._gradient_minibatch_size
                    # Use LLM to compute again action probabilities and value
                    output = self._llm_module(
                        ["score", "value"],
                        contexts=_contexts,
                        candidates=_candidates,
                        require_grad=True,
                        minibatch_size=_batch_size,
                    )
                    scores = scores_stacking([_o["score"] for _o in output])
                    # print(scores,"\n\n\n\n\n")
                    probas = torch.distributions.Categorical(logits=scores)
                    values = scores_stacking([_o["value"][0] for _o in output])

                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    log_prob = probas.log_prob(
                        current_process_buffer["actions"][_start_idx:_stop_idx]
                    )  # Use logprobs from dist as they were normalized
                    ratio = torch.exp(
                        log_prob
                        - current_process_buffer["logprobs"][_start_idx:_stop_idx]
                    )
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                    if (
                        i == 0
                        and step == 0
                        and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1))
                    ):
                        logging.warning("PPO ratio != 1 !!")

                    clip_adv = (
                        torch.clamp(
                            ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]
                        )
                        * current_process_buffer["advantages"][_start_idx:_stop_idx]
                    )
                    policy_loss = -(
                        torch.min(
                            ratio
                            * current_process_buffer["advantages"][
                                _start_idx:_stop_idx
                            ],
                            clip_adv,
                        )
                    ).mean()
                    epochs_losses["policy"].append(policy_loss.detach().cpu().item())

                    # Compute value loss
                    unclipped_value_error = (
                        values - current_process_buffer["returns"][_start_idx:_stop_idx]
                    ) ** 2
                    clipped_values = current_process_buffer["values"][
                        _start_idx:_stop_idx
                    ] + torch.clamp(
                        values - current_process_buffer["values"][_start_idx:_stop_idx],
                        -kwargs["clip_eps"],
                        kwargs["clip_eps"],
                    )
                    clipped_value_error = (
                        clipped_values
                        - current_process_buffer["returns"][_start_idx:_stop_idx]
                    ) ** 2
                    value_loss = torch.max(
                        unclipped_value_error, clipped_value_error
                    ).mean()
                    epochs_losses["value"].append(value_loss.detach().cpu().item())

                    # Compute final loss
                    loss = (
                        policy_loss
                        - kwargs["entropy_coef"] * entropy
                        + kwargs["value_loss_coef"] * value_loss
                    )
                    loss = loss / gradient_accumulation_steps
                    epochs_losses["loss"].append(loss.detach().cpu().item())

                    # Backward
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._iterator_trainable_params, kwargs["max_grad_norm"]
                )
                self.optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()

        if kwargs["save_after_update"] and accelerator.process_index == 1:
            print("Saving model...")
            model_state_dict = OrderedDict(
                {k: v for k, v in self._iterator_named_trainable_params()}
            )
            torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
            torch.save(
                self.optimizer.state_dict(),
                kwargs["output_dir"] + "/optimizer.checkpoint",
            )
            print("Model saved")

        return {
            "loss": np.mean(epochs_losses["loss"]),
            "value_loss": np.mean(epochs_losses["value"]),
            "policy_loss": np.mean(epochs_losses["policy"]),
        }


def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "goal": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
    }
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
accelerator = Accelerator()
prompt_generator = [Glam_prompt_test2, prompt_maker, swap_prompt, xml_prompt, paraphrase_prompt]
lamorel_init()


obj_extractor = ObjectExtractor(use_spacy=False)

@hydra.main(config_path="config", config_name="config")
def main(config_args):
    
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    
    # init env
    
    
    # Create LLM agent
    warnings.filterwarnings("ignore")
    config_args.lamorel_args.llm_args.use_vllm= config_args.rl_script_args.name_environment!='AlfredTWEnv'
    lm_server = Caller(
        config_args.lamorel_args,
        custom_updater=PPOUpdater(
            config_args.lamorel_args.llm_args.model_type,
            config_args.rl_script_args.minibatch_size,
            config_args.rl_script_args.gradient_batch_size,
            config_args.rl_script_args.quantized_optimizer,
        ),
        custom_model_initializer=SequentialInitializer(
            [
                PeftInitializer(
                    config_args.lamorel_args.llm_args.model_type,
                    config_args.lamorel_args.llm_args.model_path,
                    config_args.rl_script_args.use_lora,
                    config_args.lamorel_args.llm_args.load_in_4bit,
                    config_args.rl_script_args.lora_r,
                    config_args.rl_script_args.lora_alpha,
                    # list(config_args.rl_script_args.lora_target_modules),
                    config_args.lamorel_args.llm_args.pre_encode_inputs,
                ),
                WeightsLoaderInitializer(config_args.rl_script_args.loading_path,config_args.rl_script_args.vlm_loading_path),
            ]
        ),
        custom_module_functions={
            "score": LogScoringModuleFn(
                config_args.lamorel_args.llm_args.model_type,
                config_args.lamorel_args.llm_args.pre_encode_inputs,
            ),
            "value": ValueHeadModuleFn(
                config_args.lamorel_args.llm_args.model_type,
                config_args.lamorel_args.llm_args.pre_encode_inputs,
            ),
        },
    )
    if config_args.rl_script_args.name_environment!='AlfredTWEnv':
        display = Display(visible=0, size=(1024, 768))
        display.start()
    config = load_config(config_args.lamorel_args.config_alfred)
    config["env"]["task_types"]=config_args.rl_script_args.task  #Previous bugged
    
    if config_args.rl_script_args.name_environment=='AlfredTWEnv':
        env = getattr(environment, config_args.rl_script_args.name_environment)(config,config_args.rl_script_args.train_eval)
        files=env.game_files.copy()
        max_episods = min(2000, int(len(files) / config_args.rl_script_args.number_envs))
        
    else:
        env = getattr(environment, config_args.rl_script_args.name_environment)(config,config_args.rl_script_args.train_eval)
        max_episods = min(2000, int(len(env.json_file_list) / config_args.rl_script_args.number_envs))
        train_env = env.init_env(batch_size=config_args.rl_script_args.number_envs)
    # init wandb
    success=[]
    eplen=[]
    #wandb.init(project=config_args.wandb_args.project, mode=config_args.wandb_args.mode,name=config_args.wandb_args.run)
    # Set up experience buffer
    buffers = [
        PPOBuffer(
            config_args.rl_script_args.steps_per_epoch
            // config_args.rl_script_args.number_envs,
            config_args.rl_script_args.gamma,
            config_args.rl_script_args.lam,
        )
        for _ in range(config_args.rl_script_args.number_envs)
    ]

    # Prepare for interaction with environment
  #  (o, infos), ep_ret, ep_len = (
  #      train_env.reset(),
  #      [0 for _ in range(config_args.rl_script_args.number_envs)],
  #      [0 for _ in range(config_args.rl_script_args.number_envs)],
  #  )

    
    not_saved = [True for _ in range(config_args.rl_script_args.number_envs)]
    history = reset_history()
    

    transitions_buffer = [[] for _ in range(config_args.rl_script_args.number_envs)]
    not_saved = [True for _ in range(config_args.rl_script_args.number_envs)]
    proj = (
        config_args.lamorel_args.llm_args.model_path.split("/")[-1]
        + "_prompt"
        + str(config_args.rl_script_args.prompt_id)
    )
    generate_prompt = prompt_generator[config_args.rl_script_args.prompt_id]
    jump=config_args.rl_script_args.number_envs
    
    #output analysis
    all_analysis = []
    for i in tqdm(range(max_episods), desc="Evaluation"):
        promptsave=[]
        imagessave=[]
        
        traj = [{} for _ in range(config_args.rl_script_args.number_envs)] # trajectory full info for analysis
        if config_args.rl_script_args.name_environment=='AlfredTWEnv':
            train_env = env.init_env(batch_size=config_args.rl_script_args.number_envs,game_files=files[i*jump:(i+1)*jump])
            (o, infos), ep_ret, ep_len = ( train_env.reset(),[0 for _ in range(config_args.rl_script_args.number_envs)],[0 for _ in range(config_args.rl_script_args.number_envs)],)
            
            infos["goal"] = [o[__i].split("\n\n")[-1] for __i in range(len(o))]
            #infos["context"] = [o[__i].split("\n\n")[0:-1] for __i in range(len(o))]
            infos["description"] = [o[__i].split("\n\n")[0:] for __i in range(len(o))]
            
            #for __i in range(len(infos["description"])):
            #        infos["description"][__i].append("")
            _goal = infos["goal"]
            _description = infos["description"]
            #_context=infos["context"]
            past_actions=["" for _ in range(config_args.rl_script_args.number_envs)]
            o, infos = get_infos(infos, config_args.rl_script_args.number_envs)  
        else:
            o, infos = train_env.reset(json_file_id=i * config_args.rl_script_args.number_envs)
            #TODO following does not work for more than 1 environments.
            env_object_info = [infos.get('extra.object_info', {})[_i] for _i in range(config_args.rl_script_args.number_envs)]  #includes object info regarding objects in the env such as visible objects
            gt_obs = [o[__i].split("\n\n")[1] for __i in range(len(o))]  #ground truth observation on reset  for analysis
            past_actions=["" for _ in range(config_args.rl_script_args.number_envs)]
            frames = train_env.get_frames()
            _frames = []
            vlm_prompt=[]
            for _i in range(frames.shape[0]):
                    _frames.append(Image.fromarray(cv2.cvtColor(frames[_i, :, :, :], cv2.COLOR_BGR2RGB)))
                    #vlm_prompt.append(f"Your Past Action:{past_actions[_i]}.Describe your Current Observation")
                    vlm_prompt.append(f"<DETAILED_CAPTION>")
            description = lm_server.generate(contexts=_frames,prompts=vlm_prompt)
            infos["description"]=[]
            infos["goal"] = [o[__i].split("\n\n")[-1] for __i in range(len(o))]
            for i in range(config_args.rl_script_args.number_envs):
                    try:
                        infos["description"].append( [description[i]['text'].split("Assistant:")[-1]]) #TODO modified and added ['text']
                    except:
                        infos["description"].append( [description[i].split("Assistant:")[-1]])
                
            _goal = infos["goal"]
            o, infos = get_infos(infos, config_args.rl_script_args.number_envs)  
        d = [False for _ in range(config_args.rl_script_args.number_envs)]
        transitions_buffer = [[] for _ in range(config_args.rl_script_args.number_envs)]
        epit=0

        #for analysis data
        for _i in range(config_args.rl_script_args.number_envs):
            traj[_i]['task'] = infos[_i]['goal']
            traj[_i]['steps'] = []
            traj[_i]['goal_entities'] = obj_extractor.extract_objects_from_text(infos[_i]['goal'])
        while not torch.all(torch.tensor(d)):
            # generate_prompt=sample(prompt_generator,1)[0]
            
            epit+=1
            possible_actions = [list(filter(lambda x: x not in ["look","inventory"]  , _i["possible_actions"])) for _i in infos]
            
            prompts = [               prompt_maker({"info":_i,"transition_buffer":_o},config_args.rl_script_args.prompt_element,config_args.rl_script_args.transitions_buffer_len) for _i, _o in zip(infos, transitions_buffer)
            ]
            promptsave.append(prompts[0])
            imagessave.append(cv2.cvtColor(train_env.get_frames()[0,:,:,:], cv2.COLOR_BGR2RGB))

            
            
            output = lm_server.custom_module_fns(
                ["score", "value"], contexts=prompts, candidates=possible_actions
            )
            scores = scores_stacking([_o["score"] for _o in output])

            proba_dist = torch.distributions.Categorical(logits=scores)
            values = scores_stacking([_o["value"][0] for _o in output])
            sampled_actions = torch.argmax(proba_dist.probs,dim=-1)
            log_probs = proba_dist.log_prob(sampled_actions)
            actions_id = sampled_actions.cpu().numpy()
            actions_command = []
            #print(proba_dist.probs)
            #print(sampled_actions)
            #print(actions_id)
            for j in range(len(actions_id)):
                command = possible_actions[j][int(actions_id[j])]
                actions_command.append(command)
                past_actions[j]=command
                transitions_buffer[j].append({"obs": infos[j]["obs"].copy(), "act": command})
                transitions_buffer[j] = transitions_buffer[j][
                    -config_args.rl_script_args.transitions_buffer_len :
                ]


            #do anlysis and store the analysis
            entities = []
            main_entities = []
            visible_obj = []
            matching_score = []
            for _i in range(config_args.rl_script_args.number_envs):
                entities.append(obj_extractor.extract_objects_from_text(infos[_i]['obs'][0]))  #objects mentioned in the vlm description
                visible_obj.append([vis_obj['objectType'] for vis_obj in env_object_info[_i]['visible_objects']]) #visible objects
                obj_extractor.calculate_overlap_score_transformer(traj[_i]['goal_entities'], visible_obj[_i])
                #main_entities.append(list(set(visible_obj[d]) & set(traj[d]['goal_entities'])))  #intersection of the goal entities and entities (vlm entities)
                matched_entities = obj_extractor.calculate_overlap_score_transformer(traj[_i]['goal_entities'], visible_obj[_i])[1]  #intersection of the goal entities and entities (vlm entities)
                main_entities.append(matched_entities)  #intersection of the goal entities and entities (vlm entities)
                matching_score.append(obj_extractor.calculate_overlap_score_transformer(main_entities[_i], entities[_i])[0])
                traj[_i]['steps'].append(
                    {
                        'vlm_desc': infos[_i]['obs'][0],
                        'text_desc': gt_obs[_i],
                        'vis_obj': visible_obj[_i],
                        'main_obj': main_entities[_i],
                        'vlm_obj': entities[_i],
                        'matching_score': matching_score[_i]
                    }
                )
            # print(transitions_buffer)
            o, r, d, infos = train_env.step(actions_command)
            gt_obs = [o[_i] for i in range(len(o))]  #ground truth textual observations for analysis
            env_object_info = [infos.get('extra.object_info', {})[_i] for _i in range(config_args.rl_script_args.number_envs)] #ground truth objects info for analysis
            #print(r,d)
            if config_args.rl_script_args.name_environment=='AlfredTWEnv':  
                infos["goal"] = _goal
                #infos["description"] = _description
                #infos["context"]=_context
                infos["description"]=[o[__i].split("\n\n")[0:] for __i in range(len(o))]
            else:
                frames = train_env.get_frames()
                _frames = []
                vlm_prompt=[]
                for _i in range(frames.shape[0]):
                    _frames.append(Image.fromarray(cv2.cvtColor(frames[_i, :, :, :], cv2.COLOR_BGR2RGB)))
                    #vlm_prompt.append(f"Your Past Action:{past_actions[_i]}.Describe your Current Observation")
                    #vlm_prompt.append(f"describe in details the image and all objects present on it<image><end_of_utterance>\nAssistant:")
                    vlm_prompt.append(f"<DETAILED_CAPTION>")
                description = lm_server.generate(contexts=_frames,prompts=vlm_prompt)
                infos["description"]=[]
                for _i in range(config_args.rl_script_args.number_envs):
                    try:
                        infos["description"].append([description[_i].split("Assistant:")[-1]])
                    except:
                        infos["description"].append( [description[i]['text'].split("Assistant:")[-1]])
                infos["goal"]=_goal
                #print(r,d,infos["won"])
            # obss,infos=get_infos(infos,config_args.rl_script_args.number_envs)
            s = infos["won"] * 1
            
            o, infos = get_infos(infos, config_args.rl_script_args.number_envs)

            s = np.multiply(s, 1)


        success += list(s)
        if s.all()==1:  #in multi environment, now only adding to eplen if all of the environments were successfull.
            eplen.append(epit)
        for _i in range(config_args.rl_script_args.number_envs):
            traj[_i]['success'] = s[_i]
        print(f"Succeed task | {_goal} | current RS | {np.mean(success)} current eplen | {np.mean(eplen)}")
        all_analysis.extend(traj)
        print("wait 5sec")
        p=f"/home/bahaduri/VIPER/outputs/success_train_txt/{_goal[0]}{int(np.mean(success)*100)}"
        if not os.path.exists(p):
            os.mkdir(p)
            file=open(f"{p}/text.txt","w")
            for idx in range(len(promptsave)):
            #promptsave.append(prompts[0])
                file.write(promptsave[idx]+"\n")
                result = Image.fromarray(imagessave[idx])
                result.save(f'{p}/{idx}.png')
            
            #imagessave.append(train_env.get_frames()[0,:,:,:])
        #print("GameFiles",files[i*jump:(i+1)*jump])
    with open("/home/bahaduri/VIPER/outputs/trajectories_train.json", "w") as f:
        json.dump(all_analysis, f, indent=4)
    print(f"all sr:{np.mean(success)},all len:{np.mean(eplen)} ")
    lm_server.close()        
        # Perform PPO update!
       


if __name__ == "__main__":
    main()
