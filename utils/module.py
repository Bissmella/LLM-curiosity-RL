from lamorel import BaseModuleFunction, BaseModelInitializer
from typing import List
import torch
from transformers import set_seed
import bitsandbytes
from torch.nn.functional import log_softmax
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input
        self.normalization = "token"

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]
                )-1  # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            #output_tokens = minibatch["decoder_input_ids"][:, end_of_context_position+1:]
            #output_tokens = minibatch["labels"][:, end_of_context_position + 1 :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
        else:
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]
                )-1  # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["decoder_input_ids"][:, end_of_context_position+1:]
            #output_tokens = minibatch["labels"][:, end_of_context_position + 1 :]
            #output_tokens = minibatch["input_ids"][:, end_of_context_position+1:] # skip pad token
        logits = log_softmax(logits, dim=-1)
        
        tokens_logprobs = (
            torch.gather(logits, 2, output_tokens[:, :, None])
            .squeeze(-1)
            .to(torch.float32)
        )  # filter with sequence tokens
        if self.normalization == "token":
            action_length = (output_tokens != 0).sum(-1)
            # print(output_tokens)
        else:
            action_length = 1
        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(
            -1
        )  # compute final sequences' probability

        minibatch_probs = minibatch_probs / action_length
        return minibatch_probs.cpu()


class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        if "hidden_size" in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map["hidden_size"]
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        #print(forward_outputs.keys())
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]
                )  # inputs are padded so all of same size

            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()


class SequentialInitializer(BaseModelInitializer):
    def __init__(self, initializers: List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers

    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)

        return model


class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path,vlm_weights_path):
        super().__init__()
        self._weights_path = weights_path
        self._vlm_weights_path = vlm_weights_path

    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {
                _k.replace("module.", ""): _v for _k, _v in loaded_ddp_dict.items()
            }
            print("im here")
            model.load_state_dict(state_dict=hf_llm_module_dict, strict= not True)
        if self._vlm_weights_path is not None:
            loaded_ddp_dict = load_file(self._vlm_weights_path + "/adapter_model.safetensors")
            hf_llm_module_dict = {
                _k.replace("module.", "").replace("base_model","_VLM_model.base_model").replace(".weight",".default.weight"): _v for _k, _v in loaded_ddp_dict.items()
            }
            #model.load_state_dict(state_dict=hf_llm_module_dict, strict=not True)

        return model


class PeftInitializer(BaseModelInitializer):
    def __init__(
        self,
        model_type,
        model_name,
        use_lora,
        use_4bit,
        r,
        alpha,
        target_modules=None,
        use_cache=True,
    ):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._target_modules = target_modules
        self._use_cache = use_cache

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_model_config(self):
        if "t5" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
        elif "bart" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
        elif "falcon" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=self._target_modules
                or [
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif (
            "opt" in self._model_name
            or "Llama" in self._model_name
            or "Mistral" in self._model_name
        ):
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules= ["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif "gpt" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif "pythia" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise NotImplementedError()

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules["_LLM_model"]
            if "_VLM_model" in model._modules.keys():
                vlm_module =model._modules["_VLM_model"]
            if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            # Init adapters #
            
            config2= LoraConfig(r=self._r,lora_alpha=self._alpha,target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "lm_head", "fc2"],task_type="CAUSAL_LM",lora_dropout=0.05,bias="none",inference_mode=False,use_rslora=True,init_lora_weights="gaussian")
            config = self._get_model_config()
            peft_model = get_peft_model(llm_module, config)
            if "_VLM_model" in model._modules.keys():
                peft_model2= get_peft_model(vlm_module, config2)
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None
            if "_VLM_model" in model._modules.keys():
                for param in peft_model2.parameters():
                #    #param.requires_grad = True
                #    if name.split(".")[-1].startswith("lora_"):
                #        if hasattr(param, "weight"):
                #            param.to(parent_module_device)
                #    else:
                #        if hasattr(param, "weight"):
                #            parent_module_device = param.weight.device
                #        else:
                #            parent_module_device = None
                    param.requires_grad = False
            if "_LLM_model" in model._modules.keys() :
                for param in peft_model.parameters():
                    if name.split(".")[-1].startswith("lora_"):
                        if hasattr(param, "weight"):
                            param.to(parent_module_device)
                    else:
                        if hasattr(param, "weight"):
                            parent_module_device = param.weight.device
                        else:
                            parent_module_device = None
                    #param.requires_grad = False

            model._modules["_LLM_model"] = peft_model
            if "_VLM_model" in model._modules.keys():
                model._modules["_VLM_model"] = peft_model2
        
        model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout)
        model._modules["_LLM_model"].config.use_cache = self._use_cache
        self._print_trainable_parameters(peft_model)
        #self._print_trainable_parameters(peft_model2)
        self._print_trainable_parameters(model)
        return model
