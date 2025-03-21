import torch
import tqdm
import np

from src.config import CONFIG

def extract_finetuned_embeddings(texts, model, tokenizer, batch_size=CONFIG.BATCH_SIZE, max_length=CONFIG.MAX_LENGTH):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Extract embeddings from the last hidden state
            hidden_states = output.hidden_states[-1]
            pooled_embeddings = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.extend(pooled_embeddings)

    return np.array(embeddings)