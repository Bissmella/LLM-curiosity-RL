import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__))
sys.path.append(str(path.parent.parent))

import gc
import math
import os
import pickle
import warnings
from collections import OrderedDict

import bitsandbytes
import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed
from utils import *

import alfworld.agents.environment as environment
from lamorel import BaseUpdater, Caller, lamorel_init


class BCUpdater(BaseUpdater):
    def __init__(
        self,
        model_type,
        minibatch_size,
        gradient_batch_size,
        quantized_optimizer,
        gradient_minibatch_size=None,
    ):
        super(BCUpdater, self).__init__()
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
        #print("weird pro",kwargs["expert_actions_id"])
        current_process_buffer["expert_actions_id"] = kwargs["expert_actions_id"][_current_batch_ids]

        epochs_losses = {"loss": [],"acc":[]}

        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for step in range(n_minibatches):
            _minibatch_start_idx = step * self._minibatch_size
            _minibatch_end_idx = min((step + 1) * self._minibatch_size, len(contexts))

            self.optimizer.zero_grad()
            gradient_accumulation_steps = math.ceil(
                (_minibatch_end_idx - _minibatch_start_idx) / self._gradient_batch_size
            )
            for accumulated_batch in range(gradient_accumulation_steps):
                _start_idx = (
                    _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                )
                _stop_idx = _minibatch_start_idx + min(
                    (accumulated_batch + 1) * self._gradient_batch_size,
                    _minibatch_end_idx,
                )

                _contexts = contexts[_start_idx:_stop_idx]
                #print(_contexts)
                _candidates = candidates[_start_idx:_stop_idx]
                if len(_contexts) == 0:
                    break
                if self._gradient_minibatch_size is None:
                    _batch_size = sum(len(_c) for _c in _candidates)
                else:
                    _batch_size = self._gradient_minibatch_size

                # Use LLM to compute again action probabilities and value
                output = self._llm_module(
                    ["score"],  # , "value"],
                    contexts=_contexts,
                    candidates=_candidates,
                    require_grad=True,
                    minibatch_size=_batch_size,
                )
                scores = scores_stacking([_o["score"] for _o in output])
                # print(scores,"\n\n\n\n\n")
                probas = torch.distributions.Categorical(logits=scores)
                sampled_actions  = torch.argmax(probas.probs,dim=-1)
                #print("weird",current_process_buffer["expert_actions_id"])
                #print("probas",probas)
                #print("weird2",current_process_buffer["expert_actions_id"][_start_idx:_stop_idx])
                log_prob = probas.log_prob(
                    current_process_buffer["expert_actions_id"][_start_idx:_stop_idx]
                )
                #print("log_prob",log_prob)
                loss = -log_prob.mean()  # Negative Log Likelihood

                epochs_losses["loss"].append(loss.detach().cpu().item())
                epochs_losses["acc"].append((sampled_actions==current_process_buffer["expert_actions_id"][_start_idx:_stop_idx])*1)
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
            "acc": np.mean(epochs_losses["acc"]),
        }


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def collate_fn(batch):
    res = {"context":[],"candidates":[],"expert_actions_id":[]}
    
    for element in batch:
        res["context"].append(element[0])
        res["candidates"].append(element[1])
        res["expert_actions_id"].append(element[2])
    return res


prompt_generator = [prompt_maker,prompt_maker,Glam_prompt, swap_prompt, xml_prompt, paraphrase_prompt]
lamorel_init()
accelerator = Accelerator()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def Create_data(Path,fprompt,element,nbtransition):
    data=[]
    for episode in tqdm(os.listdir(Path)):
      try:
        pkl=pickle.load(open(Path+"/"+episode,'rb'))
        for i in range(len(pkl)):
          if pkl[i]['expert_action'][0]!='look' and pkl[i]['expert_action'][0]!='inventory':  
            data.append((fprompt(pkl[i],element,nbtransition,limite=-1),pkl[i]["info"]["possible_actions"],pkl[i]['expert_actions_id']))
      except:
          pass
      
             
    return data

@hydra.main(config_path="config", config_name="config")
def main(config_args):
    # REMOVE WARNINGS
    warnings.filterwarnings("ignore")

    # RANDOM SEED
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # INIT EVAL ENV
    config = load_config(config_args.lamorel_args.config_alfred)
    config["env"]["task_types"] = [config_args.rl_script_args.task]

    env = getattr(environment, config_args.rl_script_args.name_environment)(
        config, train_eval="eval_in_distribution"
    )

    
    # CREATE DATALOADER
    print(config_args.rl_script_args.prompt_element)
    file_path = config_args.rl_script_args.path_data
    data=Create_data(file_path,prompt_maker, config_args.rl_script_args.prompt_element,config_args.rl_script_args.transitions_buffer_len)

    dataset = CustomDataset(data)

    print(f"\n Train on {len(dataset)} expert transitions \n")

    dataloader = DataLoader(
        dataset,
        batch_size=1000,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    # CREATE LLM AGENT
    lm_server = Caller(
        config_args.lamorel_args,
        custom_updater=BCUpdater(
            config_args.lamorel_args.llm_args.model_type,
            config_args.rl_script_args.minibatch_size,
            config_args.rl_script_args.gradient_batch_size,
            config_args.rl_script_args.quantized_optimizer,
            config_args.rl_script_args.gradient_minibatch_size
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
            )
        },
    )

    generate_prompt = prompt_generator[config_args.rl_script_args.prompt_id]

    # INIT WANDB
    wandb.init(
        project=config_args.wandb_args.project,
        mode=config_args.wandb_args.mode,
        name=config_args.wandb_args.run,
    )
    game_files=env.game_files.copy()
    # TRAIN WITH BC
    for epoch in range(config_args.rl_script_args.epochs):
        # SAVE MODEL
        save_model_and_history = (
            epoch % config_args.rl_script_args.save_freq == 0
            or epoch == config_args.rl_script_args.epochs - 1
        ) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq
        saving_path = (
            f"{config_args.rl_script_args.output_dir}/epochs_{start_epoch}-{epoch}"
        )
        if save_model_and_history:
            os.makedirs(saving_path, exist_ok=True)
        loading_path = (
            config_args.rl_script_args.loading_path
            if config_args.rl_script_args.loading_path is not None
            else ""
        )

        # EVALUATE
        jump=1
        if not True:
            transitions_buffer = [
                [] for _ in range(config_args.rl_script_args.number_eval_envs)
            ]
            
            max_episods = min(50,len(game_files))
            success = []
            episode_length = []
            for i in tqdm(range(max_episods), desc="Evaluation"):
                train_env = env.init_env(batch_size=config_args.rl_script_args.number_eval_envs,game_files=game_files[i*jump:(i+1)*jump])

                o, infos = train_env.reset()
    
            # TO BE ADAPTED TO  VLM
                infos["goal"] = [o[__i].split("\n\n")[-1] for __i in range(len(o))]
                infos["description"] = [o[__i].split("\n\n")[0:] for __i in range(len(o))]
                
                _goal = infos["goal"]
                
                o, infos = get_infos(infos, config_args.rl_script_args.number_eval_envs)
    
                d = [False for _ in range(config_args.rl_script_args.number_eval_envs)]
                timestep = np.zeros(config_args.rl_script_args.number_eval_envs, dtype=int)
    
                
    
                while not torch.all(torch.tensor(d)):
                    timestep += np.ones_like(d, dtype=int) - np.array(d, dtype=int)
                # generate_prompt=sample(prompt_generator,1)[0]
                    #dico={"info":infos,"transition_buffer":transitions_buffer}
                    
                    possible_actions = [_i["possible_actions"] for _i in infos]
                    prompts = [prompt_maker({"info":_i,"transition_buffer":_o},config_args.rl_script_args.prompt_element,config_args.rl_script_args.transitions_buffer_len) for _i, _o in zip(infos, transitions_buffer)]
                    #print(prompts[0],"\n\n")
                # print(prompts)
                    output = lm_server.custom_module_fns(["score"], contexts=prompts, candidates=possible_actions)
                    scores = scores_stacking([_o["score"] for _o in output])
    
                    proba_dist = torch.distributions.Categorical(logits=scores)
                    sampled_actions  = torch.argmax(proba_dist.probs,dim=-1)
                    actions_id = sampled_actions.cpu().numpy()
                    actions_command = []
                    for j in range(len(actions_id)):
                        command = possible_actions[j][int(actions_id[j])]
                        actions_command.append(command)
                        transitions_buffer[j].append(
                        {"obs": infos[j]["obs"].copy(), "act": command}
                    )
                        transitions_buffer[j] = transitions_buffer[j][
                        -config_args.rl_script_args.transitions_buffer_len :
                    ]
    
                    o, r, d, infos = train_env.step(actions_command)
    
                # TO BE ADAPTED TO VLM
                    infos["goal"] = _goal
                    infos["description"] = [o[__i].split("\n\n")[0:] for __i in range(len(o))]
                    
                    s = infos["won"] * 1
                    o, infos = get_infos(infos, config_args.rl_script_args.number_eval_envs)
    
                    s = np.multiply(r, 1)
                success += list(s)
                episode_length += list(timestep)
                print(success)
    
            print(
                f"Evaluation ({config_args.rl_script_args.number_eval_envs} envs) | SR | {np.mean(success)} | Episode length | {np.mean(episode_length)}"
            )
    
            wandb.log({"succes rate": np.mean(success)})

        all_losses = []
        all_acc=[]
        # TRAINING
        for b, batch in tqdm(
            enumerate(dataloader),
            total=len(dataset) // 1000 + 1,
            desc=f"Epoch {epoch}/{config_args.rl_script_args.epochs}: Iterate on expert data",
        ):
            # UPDATE MODEL
            try:
                update_results = lm_server.update(
                batch["context"],  # context
                batch["candidates"],  # candidates
                expert_actions_id=torch.Tensor(batch["expert_actions_id"]),
                lr=config_args.rl_script_args.lr,
                clip_eps=config_args.rl_script_args.clip_eps,
                entropy_coef=config_args.rl_script_args.entropy_coef,
                value_loss_coef=config_args.rl_script_args.value_loss_coef,
                max_grad_norm=config_args.rl_script_args.max_grad_norm,
                use_all_params_for_optim=config_args.rl_script_args.use_all_params_for_optim,
                ppo_epochs=config_args.rl_script_args.ppo_epochs,
                save_after_update=save_model_and_history,
                output_dir=saving_path,
                loading_path=loading_path
            )
            #print("\n\n\n\n","loss",np.mean([_r["loss"] for _r in update_results]))
                wandb.log({"loss": np.mean([_r["loss"] for _r in update_results])})
                wandb.log({"acc": np.mean([_r["acc"] for _r in update_results])})
                all_losses.append(np.mean([_r["loss"] for _r in update_results]))
                all_acc.append(np.mean([_r["acc"] for _r in update_results]))
                print(all_acc)
            except:
                pass

        avg_loss = np.mean(all_losses)
        avg_acc = np.mean(all_acc)
        print(f"Epoch {epoch} | BC loss: {avg_loss}| BC acc: {avg_acc} ","\n\n\n")

    lm_server.close()
    exit()


if __name__ == "__main__":
    main()