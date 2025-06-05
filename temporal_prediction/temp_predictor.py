from transformers import T5Tokenizer, T5Model
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset




class TrajectoryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])


class Temp_predictor():
    def __init__(self, accelerator, optimizer, ):

        """
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5Model.from_pretrained("t5-small")
        """
        self.tokenizer = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model = T5Model.from_pretrained("t5-small")



    def preprocess_trajectories(self, trajectory,):
        input_seq= None
        target_seq= None
        return input_seq, target_seq
    

    def update_temp_predictor(self, buffer):
        train_data = [self.preprocess_trajectory(traj) for traj in buffer]

        model_inputs = self.tokenizer(
            [inp for inp, _ in train_data],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            [tgt for _, tgt in train_data],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]

        dataset = TrajectoryDataset(model_inputs)

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