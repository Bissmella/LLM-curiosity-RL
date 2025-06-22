from transformers import T5TokenizerFast, T5Model
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Config
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import warnings
import numpy as np
import gc
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import List, Optional, Tuple, Union
from collections import deque
ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]



def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class TempPredictorModel(T5ForConditionalGeneration):
    def __init__(self,  config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.LongTensor] =None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
            `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        if labels is not None:
            if loss_weights is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss_raw = loss * loss_weights.view(-1)
                loss = torch.mean(loss_raw)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss_raw=None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), loss_raw

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     encoder_outputs=None,
    #     decoder_input_ids=None,
    #     decoder_attention_mask=None,
    #     decoder_past_key_value_states=None,
    #     use_cache=None,
    #     labels=None,
    #     inputs_embeds=None,
    #     decoder_inputs_embeds=None,
    #     head_mask=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     loss_weights=None,
    #     **kwargs
    # ):
    #     r"""
    #     labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
    #         Labels for computing the sequence classification/regression loss.
    #         Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
    #         All labels set to ``-100`` are ignored (masked), the loss is only
    #         computed for labels in ``[0, ..., config.vocab_size]``
    #     kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
    #         Used to hide legacy arguments that have been deprecated.

    # Returns:
    #     :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
    #     loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
    #         Classification loss (cross entropy).
    #     prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
    #         Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    #         If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
    #     decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
    #         Contains pre-computed key and value hidden-states of the attention blocks.
    #         Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
    #         Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
    #     hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
    #         Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
    #         of shape :obj:`(batch_size, sequence_length, hidden_size)`.

    #         Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    #     attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
    #         Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
    #         :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

    #         Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
    #         heads.

    # Examples::

    #     >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

    #     >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    #     >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
    #     >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
    #     >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    #     >>> loss, prediction_scores = outputs[:2]

    #     >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    #     >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
    #     >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
    #     >>> outputs = model.generate(input_ids)
    #     """

    #     if "lm_labels" in kwargs:
    #         warnings.warn(
    #             "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
    #             DeprecationWarning,
    #         )
    #         labels = kwargs.pop("lm_labels")
    #     assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    #     use_cache = use_cache if use_cache is not None else self.config.use_cache

    #     # Encode if needed (training, first prediction pass)
    #     if encoder_outputs is None:
    #         # Convert encoder inputs in embeddings if needed
    #         encoder_outputs = self.encoder(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             inputs_embeds=inputs_embeds,
    #             head_mask=head_mask,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #         )

    #     hidden_states = encoder_outputs[0]

    #     if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
    #         # get decoder inputs from shifting lm labels to the right
    #         decoder_input_ids = self._shift_right(labels)

    #     # If decoding with past key value states, only the last tokens
    #     # should be given as an input
    #     if decoder_past_key_value_states is not None:
    #         assert labels is None, "Decoder should not use cached key value states when training."
    #         if decoder_input_ids is not None:
    #             decoder_input_ids = decoder_input_ids[:, -1:]
    #         if decoder_inputs_embeds is not None:
    #             decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    #     # Decode
    #     decoder_outputs = self.decoder(
    #         input_ids=decoder_input_ids,
    #         attention_mask=decoder_attention_mask,
    #         inputs_embeds=decoder_inputs_embeds,
    #         past_key_value_states=decoder_past_key_value_states,
    #         encoder_hidden_states=hidden_states,
    #         encoder_attention_mask=attention_mask,
    #         head_mask=head_mask,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #     )

    #     # insert decoder past at right place
    #     # to speed up decoding
    #     if use_cache is True:
    #         past = ((encoder_outputs, decoder_outputs[1]),)
    #         decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

    #     sequence_output = decoder_outputs[0]
    #     # Rescale output before projecting on vocab
    #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    #     sequence_output = sequence_output * (self.model_dim ** -0.5)
    #     lm_logits = self.lm_head(sequence_output)

    #     decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
    #     if labels is not None:
    #         if loss_weights is not None:
    #             loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    #             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    #             loss_raw = loss * loss_weights.view(-1)
    #             loss = torch.mean(loss_raw)
    #         else:
    #             loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    #             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    #             loss_raw=None
            
    #         # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
    #         decoder_outputs = (loss,) + (loss_raw,) + decoder_outputs

    #     return decoder_outputs + encoder_outputs




class TrajectoryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


class CustomCollator:
    def __init__(self, tokenizer, model):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def __call__(self, batch):
        # Separate the weights before calling the base collator
        weights = [torch.tensor(example["weights"], dtype=torch.float) for example in batch]

        # Remove weights from examples to avoid issues with the base collator
        for example in batch:
            del example["weights"]

        # Use Huggingface's default collator to pad everything else
        batch_out = self.base_collator(batch)

        # Pad weights to match the padded labels
        padded_weights = pad_sequence(weights, batch_first=True, padding_value=1.0)  # default neutral weight

        batch_out["weights"] = padded_weights

        return batch_out


class Temp_predictor():
    def __init__(self, device, epochs=4, temp_model_lr=3e-4, buff_size=160):

        """
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5Model.from_pretrained("t5-small")
        """
        self.act_sep_token = '<ACT_SEP>'  #special action seperator token
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small", extra_special_tokens={"act_token": self.act_sep_token})
        config = T5Config.from_pretrained("t5-small")
        self.model = TempPredictorModel(config)
        #self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.temp_model_optimizer = torch.optim.Adam(self.model.parameters(), lr = temp_model_lr)
        self.device = device
        #self.accelerator = accelerator
        self.epochs = epochs
        
        self.tokenizer.add_special_tokens({'act_token': self.act_sep_token})  #action seperator token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.act_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.act_sep_token)
        #self.temp_model_optimizer = self.accelerator.prepare(self.temp_model_optimizer)
        self.model.to(self.device)

        #trajectories buffer
        self.input_seq_buffer = deque(maxlen=buff_size)
        self.target_seq_buffer = deque(maxlen=buff_size)


    def preprocess_trajectories(self, buffer_goals, buffer_trajLen, buffer_actions, terminals):
        input_seq = []
        target_seq = []
        counter =0
        for i, len in enumerate(buffer_trajLen):
            acts = buffer_actions[counter: len]
            counter += len
            if terminals[i]:
                self.input_seq_buffer.append(buffer_goals[i])
                flattened = f" {self.act_sep_token} ".join(acts)
                self.target_seq_buffer.append(flattened)
        # input_seq= None
        # target_seq= None
        return list(self.input_seq_buffer), list(self.target_seq_buffer)
    
    def preprocess_data(self, examples):
        model_inputs = self.tokenizer(examples["input"], truncation=True, padding=False)
        labels = self.tokenizer(examples["target"], return_offsets_mapping=True, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs, labels["offset_mapping"]

    def update_model(self, buffer_goals, buffer_trajLen, buffer_actions, terminals):
        self.model.train()
        input_seq, target_seq = self.preprocess_trajectories(buffer_goals, buffer_trajLen, buffer_actions, terminals) #train_data
        model_inputs = self.tokenizer(
            input_seq, #[inp for inp, _ in train_data],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels = self.tokenizer(
            target_seq, #[tgt for _, tgt in train_data],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        token_offsets = labels["offset_mapping"]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs = self.assign_token_weights2(target_seq, model_inputs, token_offsets)
        dataset = TrajectoryDataset(model_inputs)
        data_collator = CustomCollator(tokenizer=self.tokenizer, model=self.model)
        train_loader = DataLoader(dataset, batch_size=64, collate_fn=data_collator, shuffle=True)
        #TODO set the max new tokens of the model to max_tokens of the policy LLM * max_steps of the policy
        #train_loader = self.accelerator.prepare(train_loader)
        info = {}
        info_list = []
        for _ in tqdm(range(self.epochs)):
            self.temp_model_optimizer.zero_grad()
            epoch_loss= 0
            correct = 0
            total = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                input_attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss_weights = batch['weights'].to(self.device)
                #labels_attention_mask = label['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=input_attention_mask,
                                    encoder_outputs=None,
                                    labels= labels,
                                    loss_weights = loss_weights,
                                    # decoder_input_ids=labels_input_ids,
                                    # decoder_attention_mask=labels_attention_mask,
                                    )
                loss, logits = outputs[0].loss, outputs[0].logits  #TODO  3 is placeholder for the logits #outputs.loss, outputs.logits
                #self.accelerator.backward(loss)
                loss.backward()
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            info_list.append({"temp_predictor.loss": avg_epoch_loss, "temp_predictor.acc": train_accuracy})
            #self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.temp_model_optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()

        info.update(dict_mean(info_list))
        return info


            


    def eval_model(self, sequence):
        """
        To be used for both validation of the temporal predictor model and also for novelty score calculation.
        """
        self.model.eval()
        eval_data = self.preprocess_trajectories(sequence)
        model_inputs = self.tokenizer(
            [inp for inp, _ in eval_data],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels = self.tokenizer(
            [tgt for _, tgt in eval_data],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        token_offsets = labels["offset_mapping"]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs = self.assign_token_weights2([tgt for _, tgt in eval_data], model_inputs, token_offsets)
        eval_dataset = TrajectoryDataset(model_inputs)
        data_collator = CustomCollator(tokenizer=self.tokenizer, model=self.model)
        eval_loader = DataLoader(eval_dataset, batch_size=64, collate_fn=data_collator, shuffle=False)
        
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    input_attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    loss_weights = batch['weights'].to(self.device)
                    #labels_attention_mask = label['attention_mask'].to(self.device)

                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=input_attention_mask,
                                        encoder_outputs=None,
                                        # decoder_input_ids=labels_input_ids,
                                        # decoder_attention_mask=labels_attention_mask,
                                        )
                    loss, logits = outputs[0].loss, outputs[0].logits
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    mask = labels != -100
                    correct += ((preds == labels) & mask).sum().item()
                    total += mask.sum().item()
        avg_val_loss = val_loss/ len(eval_loader)
        val_accuracy = correct/total
        
        #
        return loss   #higher = more novel
    
    def compute_novelty(self, input, target):
        """
        computes novetly for one trajectory
        input: a string, mainly the trajectory's goal
        target: list of strings, mainly the trajectory's actions
        """
        self.model.eval()
        target = f" {self.act_sep_token} ".join(target)
        model_inputs = self.tokenizer(
            [input],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels_ = self.tokenizer(
            [target],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        
        #creating slices of actions
        #TODO check the labels might need an indexing like labels[0]
        act_sep_indices = [i for i, x in enumerate(labels_["input_ids"][0]) if x == self.act_sep_token_id]
        start_indices = [0] + [i + 1 for i in act_sep_indices]
        end_indices = act_sep_indices + [len(labels_["input_ids"][0])-1]
        # Now pair them into slices
        action_slices = [slice(start, end) for start, end in zip(start_indices, end_indices)]

        token_offsets = labels_["offset_mapping"]
        model_inputs["labels"] = labels_["input_ids"]
        model_inputs = self.assign_token_weights2([target], model_inputs, token_offsets)
        #TODO put a breakpoint here and check if no index [0] needed after the model_inputs
        #breakpoint()
        input_ids = torch.tensor(model_inputs["input_ids"]).to(self.device)
        input_attention_mask = torch.tensor(model_inputs['attention_mask']).to(self.device)
        loss_weights = torch.tensor(model_inputs['weights']).to(self.device)
        labels = torch.tensor(model_inputs['labels']).to(self.device)
        outputs = self.model(input_ids=input_ids,
                                        attention_mask=input_attention_mask,
                                        encoder_outputs=None,
                                        labels = labels,
                                        loss_weights=loss_weights,
                                        # decoder_input_ids=labels_input_ids,
                                        # decoder_attention_mask=labels_attention_mask,
                                        )
        
        loss_raw = outputs[1]
        
        loss_sliced = [loss_raw[s] for s in action_slices]
        action_loss_normalized = np.array([s.mean().item() for s in loss_sliced])
        epsilon = 1e-8
        final_loss = (action_loss_normalized - np.min(action_loss_normalized))/(np.max(action_loss_normalized) - np.min(action_loss_normalized) + epsilon)
        return final_loss



    def assign_token_weights2(self, text, model_inputs, offsets):
        sentence_spans = []
        model_inputs['weights'] = []
        for sentence in text:
            cursor = 0
            word_span = []
            for word in sentence.split():
                word_span.append((cursor, cursor + len(word)))
                cursor += len(word) + 1
            sentence_spans.append(word_span)
        
        for i, word_spans in enumerate(sentence_spans):
            offset = offsets[i]
            offset_counter = 0
            weights = []
            for j, word_span in enumerate(word_spans):
                #if offset[offset_counter][1] == word_span[1]
                counter = 0
                while (offset_counter + counter < len(offset)) and offset[offset_counter + counter][1] <= word_span[1]:
                    if text[i][word_span[0]: word_span[1]] in ALF_ACTION_LIST:
                        #breakpoint()
                        weights.append(2)
                    else:
                        weights.append(1)
                    counter += 1
                offset_counter += counter
            while len(weights) < len(offset):
                weights.append(1)
            model_inputs['weights'].append(weights)
        return model_inputs
    
    def assign_token_weights(self, text, model_inputs, offsets):
        model_inputs['weights'] = []

        for i, sentence in enumerate(text):
            cursor = 0
            word_spans = []
            for word in sentence.split():
                word_spans.append((cursor, cursor + len(word)))
                cursor += len(word) + 1  # +1 for space

            offset = offsets[i]
            offset_counter = 0
            weights = []

            for word_span in word_spans:
                start, end = word_span
                token_weights = []

                # Check if this word is an action
                is_action = sentence[start:end] in ALF_ACTION_LIST
                #breakpoint()
                # Assign weights to all tokens that belong to this word
                counter = 0
                #breakpoint()
                while (offset_counter + counter < len(offset)) and (offset[offset_counter + counter][1] <= end):
                    #breakpoint()
                    if offset[offset_counter + counter] == (0, 0):  # special token
                        token_weights.append(1)
                    else:
                        token_weights.append(2 if is_action else 1)
                    #breakpoint()
                    counter += 1

                # Add final token that ends the word
                # if (offset_counter + counter < len(offset)):
                #     token_weights.append(2 if is_action else 1)
                #     counter += 1

                offset_counter += counter
                weights.extend(token_weights)

            # Fill with default weight if any tokens are left (e.g. punctuation or special tokens)
            while len(weights) < len(offset):
                weights.append(1)

            model_inputs['weights'].append(weights)

        return model_inputs
    

                    
