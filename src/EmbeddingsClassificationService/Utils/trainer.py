import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from src.config import ANNConfig

class Training:
    def __init__(self, model, num_epochs=ANNConfig.NUM_EPOCHS, batch_size=ANNConfig.BATCH_SIZE):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=ANNConfig.LR_MAX, weight_decay=ANNConfig.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.best_val_loss = float("inf")

    def prepare_data(self, X_train, y_train, X_val, y_val):
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.X_test = X_val
        self.y_test = y_val

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch") as pbar:
                for X_batch, y_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            val_loss = self.evaluate()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {epoch_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_ann.pth")
                print(f"Model saved at epoch {epoch+1} (Best Val Loss: {val_loss:.4f})")

        self.load_best_model()

    def evaluate(self):
        import numpy as np
        from sklearn.metrics import log_loss

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(self.X_test)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)  # Apply softmax
            probabilities_np = probabilities.cpu().numpy()


        log_loss = log_loss(self.y_test, probabilities_np)

        print(f"\nLog Loss: {log_loss:.4f}")
        torch.save(self.model.state_dict(), ANNConfig.CACHE_DIR)

    def load_best_model(self):
        self.model.load_state_dict(torch.load("best_ann.pth"))
        print("Best model loaded from checkpoint.")
