import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, BigBirdForSequenceClassification
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ann_classifier import ANNClassifier
from data_prep import DataPrep

MODEL_PATH = "./bert-trained"
MODEL_SAVE_PATH = "ann_classifier.pth"

def process_csv(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

def extract_finetuned_embeddings(texts, model, tokenizer, batch_size=8, max_length=1024):
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



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
fine_tuned_model = BigBirdForSequenceClassification.from_pretrained(MODEL_PATH, num_labels = 3)
fine_tuned_model.eval()

df = DataPrep.perform("dataset/train.csv")
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


X_train = extract_finetuned_embeddings(train_df["text"].tolist(), fine_tuned_model, tokenizer, batch_size=16)
X_test = extract_finetuned_embeddings(test_df["text"].tolist(), fine_tuned_model, tokenizer, batch_size=16)

y_train = train_df["label"].values
y_test = test_df["label"].values



X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = ANNClassifier(input_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)


X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

num_epochs = 100
batch_size = 32
best_val_loss = float("inf")

# Create DataLoaders for batching
train_dataset = TensorDataset(X_train_final, y_train_final)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    # val loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()

    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_ann.pth")
        print(f"Model saved at epoch {epoch+1} (Best Val Loss: {val_loss:.4f})")

model.load_state_dict(torch.load("best_ann.pth"))
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor.to(device))
    probabilities = torch.nn.functional.softmax(predictions, dim=1)  # Apply softmax
    predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

accuracy = np.mean(predicted_labels == y_test)
print(f"\nAccuracy: {accuracy:.4f}")

for i in range(2):
    print(f"Sample {i+1}:")
    print(f"Probabilities: {probabilities_np[i]}")
    print(f"Predicted Label: {predicted_labels[i]}\n")

from sklearn.metrics import log_loss

log_loss = log_loss(y_test, probabilities_np)

print(f"\nLog Loss: {log_loss:.4f}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
