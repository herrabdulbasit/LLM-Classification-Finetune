from src.config import CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def load_model(model_path=None, full_finetune=False):
    if model_path is not None:
        MODEL_NAME = model_path
    else:
        MODEL_NAME = CONFIG.MODEL_NAME

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    except:
        print("Loading Model Failed, Check logs for more info")
        #TODO: Add Logging
        return None, None


   
    if not full_finetune:
        lora_config = LoraConfig(
            r=CONFIG.LORA_RANK,
            lora_alpha=CONFIG.LORA_ALPHA,
            lora_dropout=0.0,
            target_modules=CONFIG.LORA_MODULES,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = get_peft_model(model, lora_config)


    tokenizer.pad_token = tokenizer.eos_token

    # If EOS does not exist, manually set [PAD]
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer