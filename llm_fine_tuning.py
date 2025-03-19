from transformers import DebertaV2Tokenizer, AutoTokenizer
import pandas as pd 
import numpy as np
from config import CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from hf_dataset import ResponseClassificationDataset
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import wandb
import torch
import torch.nn as nn
from lion_pytorch import Lion
from data_prep import DataPrep
from trainer import MyTrainer


def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    return [len(t) for t in input_ids]

def load_data(tokenizer):
    from sklearn.model_selection import train_test_split

    train = DataPrep("dataset/train.csv").perform()
    train.loc[:, 'token_count'] = get_token_lengths(train['text'], tokenizer)
    train = train[train["token_count"]<= 1024]
    print('Total train samples: ', len(train))
    train, eval_df = train_test_split(train, test_size=0.1, random_state=42)

    return train, eval_df


def get_model():
    MODEL_NAME = CONFIG.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)


    #Uncomment for LoRA fine-tuning. Leave it commented for a full finetune.

    # LoRA configuration
    lora_config = LoraConfig(
        r=CONFIG.LORA_RANK,
        lora_alpha=CONFIG.LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=CONFIG.LORA_MODULES,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Applying LoRA to the model
    model = get_peft_model(model, lora_config)


    tokenizer.pad_token = tokenizer.eos_token

    # If EOS does not exist, manually set [PAD]
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer




def main():
    model, tokenizer = get_model()
    train, eval_df = load_data(tokenizer)

    train_dataset = ResponseClassificationDataset(train.sample(n=5000, random_state=42), tokenizer)
    eval_dataset = ResponseClassificationDataset(eval_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=14, shuffle=False)

    trainer = MyTrainer(model, train_dataset, eval_dataset, tokenizer)
    trainer.train()
