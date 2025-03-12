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


def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    return [len(t) for t in input_ids]

def load_data(tokenizer):
    train = pd.read_csv('dataset/train.csv')

    def process(input_str):
        stripped_str = input_str.strip('[]')
        sentences = [s.strip('"') for s in stripped_str.split('","')]
        return  ' '.join(sentences)

    train.loc[:, 'prompt'] = train['prompt'].apply(process)
    train.loc[:, 'response_a'] = train['response_a'].apply(process)
    train.loc[:, 'response_b'] = train['response_b'].apply(process)

    indexes = train[(train.response_a == 'null') & (train.response_b == 'null')].index
    train.drop(indexes, inplace=True)
    train.reset_index(inplace=True, drop=True)

    train["text"] = (
        "[USER PROMPT]: " + train["prompt"] + "\n\n"
        "[MODEL A]: " + train["response_a"] + "\n\n"
        "[MODEL B]: " + train["response_b"] + "\n\n"
        "Which response is better? (0) Model A, (1) Model B, (2) Tie"
    )

    train.loc[:, 'token_count'] = get_token_lengths(train['text'], tokenizer)
    train.loc[:, 'label'] = np.argmax(train[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)
    train = train[train["token_count"]<= 1024]

    print(f"Total {len(indexes)} Null response rows dropped")
    print('Total train samples: ', len(train))

    from sklearn.model_selection import train_test_split

    train, eval_df = train_test_split(train, test_size=0.1, random_state=42)

    return train, eval_df


def get_model():
    MODEL_NAME = CONFIG.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    '''
    Uncomment for LoRA fine-tuning. Leave it commented for a full finetune.

    # LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=CONFIG.LORA_MODULES,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Applying LoRA to the model
    model = get_peft_model(model, lora_config)
    '''

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

    def get_lion_optimizer(model):
        return Lion(model.parameters(), lr=5e-5, weight_decay=0.0)

    # Modify Trainer to use Lion optimizer
    class LionTrainer(Trainer):
        def create_optimizer(self):
            self.optimizer = get_lion_optimizer(self.model)
            return self.optimizer

    wandb.init(project="llm-finetuning-llama-new-run")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.0,
        logging_dir="./logs",
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adafactor",
        report_to="wandb",
        logging_steps=10,
        lr_scheduler_type="cosine",
        max_grad_norm=2.0,
        warmup_ratio=0.1,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = LionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    try:
        trainer.train(resume_from_checkpoint=False)
        output_dir = "./googe-berta-trained"
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        wandb.finish()
    except KeyboardInterrupt:
        wandb.finish()

