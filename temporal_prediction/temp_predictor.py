from transformers import T5Tokenizer, T5Model
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq


ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

class TempPredictorModel(T5ForConditionalGeneration):
    def __init__(self, ):
        super().__init__()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
        >>> loss, prediction_scores = outputs[:2]

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
        >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache

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
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs




class TrajectoryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


class Temp_predictor():
    def __init__(self, accelerator, optimizer, temp_model_lr, device):

        """
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5Model.from_pretrained("t5-small")
        """
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.temp_model_optimizer = torch.optim.Adam(self.model.parameters(), lr = temp_model_lr)
        self.device = device
        self.accelerator = accelerator



    def preprocess_trajectories(self, trajectory,):
        input_seq= None
        target_seq= None
        return input_seq, target_seq
    
    def preprocess_data(self, examples):
        model_inputs = self.tokenizer(examples["input"], truncation=True, padding=False)
        labels = self.tokenizer(examples["target"], return_offsets_mapping=True, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs, labels["offset_mapping"]

    def update_temp_predictor(self, buffer):
        train_data = [self.preprocess_trajectory(traj) for traj in buffer]

        model_inputs = self.tokenizer(
            [inp for inp, _ in train_data],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            [tgt for _, tgt in train_data],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors="pt"
        )
        token_offsets = labels["offset_mapping"]
        model_inputs["labels"] = labels["input_ids"]

        dataset = TrajectoryDataset(model_inputs)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        train_loader = DataLoader(dataset, batch_size=64, collate_fn=data_collator, shuffle=True)
        self.model.zero_grad()
        epoch_loss= 0
        correct = 0
        total = 0

        for input, label in train_loader:
            input_ids = input['input_ids'].to(self.device)
            input_attention_mask = input['attention_mask'].to(self.device)
            labels_input_ids = label['input_ids'].to(self.device)
            labels_attention_mask = label['attention_mask'].to(self.device)

            outputs = self.model(input_ids=input_ids,
                                attention_mask=input_attention_mask,
                                encoder_outputs=None,
                                decoder_input_ids=labels_input_ids,
                                decoder_attention_mask=labels_attention_mask,)
            loss, logits = outputs.loss, outputs.logits

            
        training_args = TrainingArguments(
                    output_dir="./t5_trajectory",
                    per_device_train_batch_size=8,
                    num_train_epochs=10,
                    logging_dir="./logs",
                    save_steps=500,
                    save_total_limit=2,
                )
        
        trainer = Trainer(
            model= self.model,
            args = training_args,
            train_dataset = dataset
            )
        
        trainer.train()


    def compute_novelty(self, sequence):
        input_seq, target_seq = self.preprocess_trajectory(sequence)
        input_ids = self.tokenizer(input_seq, return_tensors="pt", truncation=True).input_ids
        target_ids = self.tokenizer(target_seq, return_tensors="pt", truncation=True).input_ids

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss.item()
        return loss   #higher = more novel
    

    def assign_token_weights(self, text, important_words, tokenizer):
        encoding = tokenizer(text, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        words = text.split()

        # Create character span lookup for each word
        word_spans = []
        cursor = 0
        for word in words:
            start = text.find(word, cursor)
            end = start + len(word)
            word_spans.append((start, end))
            cursor = end

        # Map token offset to word weight
        token_weights = []
        for (start, end) in offsets:
            matched = None
            for i, (w_start, w_end) in enumerate(word_spans):
                if start >= w_start and end <= w_end:
                    matched = words[i]
                    break
            if matched and matched in important_words:
                token_weights.append(2.0)
            else:
                token_weights.append(1.0)

        return token_weights 