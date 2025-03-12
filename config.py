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
    NUM_EPOCHS = 1
    BATCH_SIZE = 8
    DROPOUT = 0.05 
    MODEL_NAME = "allenai/longformer-base-4096"
    SEED = 42
    MAX_LENGTH = 1024 
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5 
    NUM_LABELS = 3 
    LORA_RANK = 4
    LORA_ALPHA = 8
    CACHE_DIR = "/media/data/MODELS"
    LORA_MODULES = target_modules