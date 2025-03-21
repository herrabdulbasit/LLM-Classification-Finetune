target_modules = [
    "query",
    "key",
    "value",
    "query_global",
    "key_global",
    "value_global",
    "output.dense"
]

class CONFIG:
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    DROPOUT = 0.05 
    MODEL_NAME = "allenai/longformer-base-4096"
    SEED = 42
    MAX_LENGTH = 1024 
    WARM_UP_RATIO = 0.1
    LR_MAX = 5e-5 
    NUM_LABELS = 3 
    LORA_RANK = 32
    LORA_ALPHA = 8
    CACHE_DIR = "/media/data/MODELS"
    LORA_MODULES = target_modules
    OUTPUT_DIR = "./train_checkpoints"
    GRADIENT_ACC_STEPS = 8
    WEIGHT_DECAY=0.0
    LOGGING_DIR="./logs"
    SCHEDULER="cosine"
    METRIC_FOR_BEST_MODEL="loss"
    GREATER_IS_BETTER=False
    EVAL_STRATEGY="steps"
    EVAL_STEPS=1000
    SAVE_STRATEGY="epoch"
    WANDB_PROJECT="llm-finetuning-llama-new-run"


class ANNConfig(CONFIG):
    BATCH_SIZE=16
    NUM_EPOCHS=100
    BATCH_SIZE=32
    LR_MAX=0.0005
    WEIGHT_DECAY=1e-4
    CACHE_DIR="/home/abdulbasit/ann_models"