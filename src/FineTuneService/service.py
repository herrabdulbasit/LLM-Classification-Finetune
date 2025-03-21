from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from src.FineTuneService.ModelUtils.model_loader import load_model
from src.config import CONFIG
from src.DataService.data_loader import load_data
from src.FineTuneService.DataSetUtils.response_classification_dataset import ResponseClassificationDataset
from src.FineTuneService.DataSetUtils.inference_dataset import InferenceDataset
from src.FineTuneService.ModelUtils.trainer import MyTrainer

class FineTuneService:
    def __init__(self, mode="train", full_finetune=False):
        self.mode = mode
        self.full_finetune = full_finetune

    '''
        Loads and returns the model
        If the script is running in train mode, load the model from HuggingFace
        Otherwise loads the model from specified Cache directory where the finetuned models
        are saved for inference
    '''
    def load_model(self):
        if self.mode == 'train':
            self.model, self.tokenizer = load_model(full_finetune=self.full_finetune)
        else:
            self.model, self.tokenizer = load_model(model_path=CONFIG.CACHE_DIR, full_finetune=self.full_finetune)


    '''
        Runs a fine tine on the specified LLM Model
        Then saves the finetuned model and the tokenizer
        in the Cache Dir specified in the Project Config
    '''
    def finetune(self):
        if not self.mode == "train":
            raise ValueError("Service is not in Train Mode. Please initialize with train")
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model has not been initialized. Please initalize by calling 'load_model()'")
        
        model = self.model
        tokenizer = self.tokenizer
        train, eval_df = load_data(tokenizer)

        train_dataset = ResponseClassificationDataset(train.sample(n=5000, random_state=CONFIG.SEED), tokenizer)
        eval_dataset = ResponseClassificationDataset(eval_df, tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

        trainer = MyTrainer(model, train_dataset, eval_dataset, tokenizer)
        trainer.train()

        model.save_pretrained(CONFIG.CACHE_DIR)
        tokenizer.save_pretrained(CONFIG.CACHE_DIR)

    '''
        Loads the finetuned model, and runs the inference on the test set
        Saves the inference tests in a submission.csv file
    '''
    def run_inference(self):
        if self.mode == "train":
            raise ValueError("Service is not in Inference Mode. Please initialize with inference")
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model was not found in the directory specified or has not been loaded. Please check config.py or try calling 'load_model()'")
        
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        print("Loaded model into eval mode. Starting inference")
        _, test = load_data(tokenizer)
        test_dataset = InferenceDataset(test["text"].tolist(), tokenizer, max_length=CONFIG.MAX_LENGTH)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        all_preds = []
        device = model.device
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Generating Predictions"):
                batch = {k: v.to(device) for k, v in batch.items()}
        
                outputs = model(**batch)
                logits = outputs.logits
        
                probs = torch.nn.functional.softmax(logits, dim=-1)
        
                all_preds.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        
        
        submission_df = pd.DataFrame({
            "id": test["id"],
            "winner_model_a": all_preds[:, 0],
            "winner_model_b": all_preds[:, 1],
            "winner_tie": all_preds[:, 2]
        })
         
        submission_df.to_csv("submission.csv", index=False)