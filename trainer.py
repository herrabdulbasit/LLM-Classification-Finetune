from lion_pytorch import Lion
from transformers import Trainer, TrainingArguments
from config import CONFIG
import wandb

def get_lion_optimizer(model):
    return Lion(model.parameters(), lr=5e-5, weight_decay=0.0)

# Modify Trainer to use Lion optimizer
class LionTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = get_lion_optimizer(self.model)
        return self.optimizer
    
class MyTrainer():
    def __init__(self, model, train_df, eval_df, tokenizer):
        self.model = model
        self.train_df = train_df
        self.eval_df = eval_df
        self.tokenizer = tokenizer
        self.__init_trainer__()
    
    def train(self):
        if not self.trainer:
            raise ValueError("Trainer not initialized")
        
        wandb.init(project=CONFIG.WANDB_PROJECT)

        try:
            self.trainer.train()
            wandb.finish()
        except:
            wandb.finish()

    def training_args(self):
        return TrainingArguments(**self.__training_args__())
    
    def __init_trainer__(self):
        self.trainer = LionTrainer(
            model=self.model,
            args=self.training_args(),
            train_dataset=self.train_df,
            eval_dataset=self.eval_df,
            tokenizer=self.tokenizer,
        )

        return self.trainer

    
    def __training_args__():
        return  {
            "output_dir": CONFIG.OUTPUT_DIR,
            "evaluation_strategy": CONFIG.EVAL_STRATEGY,
            "eval_steps": CONFIG.EVAL_STEPS,
            "save_strategy": CONFIG.SAVE_STRATEGY,
            "per_device_train_batch_size": CONFIG.BATCH_SIZE,
            "per_device_eval_batch_size": CONFIG.BATCH_SIZE,
            "gradient_accumulation_steps": CONFIG.GRADIENT_ACC_STEPS,
            "num_train_epochs": CONFIG.NUM_EPOCHS,
            "learning_rate": CONFIG.LR_MAX,
            "weight_decay": CONFIG.WEIGHT_DECAY,
            "logging_dir": CONFIG.LOGGING_DIR,
            "save_total_limit": 2,
            "gradient_checkpointing": True,
            "optim": "adafactor",
            "report_to": "wandb",
            "logging_steps": 10,
            "lr_scheduler_type": CONFIG.SCHEDULER,
            "max_grad_norm": 2.0,
            "warmup_ratio": CONFIG.WARM_UP_RATIO,
            "metric_for_best_model": CONFIG.METRIC_FOR_BEST_MODEL,
            "greater_is_better": CONFIG.GREATER_IS_BETTER,
        }
    