import torch

from src.EmbeddingsClassificationService.Utils.extract_embeddings import extract_finetuned_embeddings
from src.config import ANNConfig
from src.FineTuneService.ModelUtils.model_loader import load_model
from src.DataService.data_loader import load_data
from src.EmbeddingsClassificationService.Utils.trainer import Training
from src.EmbeddingsClassificationService.Model.ann_classifier import ANNClassifier

class ANNClassifierService:

    def __init__(self):
        pass

    '''
        Loads the finetuned model from FineTuneService.
        If fails, Cache Dir defined in config.py
    '''
    def load_finetuned_model(self):
        self.ft_model, self.tokenizer = load_model(model_path=ANNConfig.CACHE_DIR)
        
        self.__check_model_presence__()

    '''
        Preps the data for ANN model training.
        returns X_train, X_test, y_train and y_tets
    '''
    def prepare_data(self):
        from sklearn.model_selection import train_test_split

        self.__check_model_presence__()

        fine_tuned_model = self.ft_model
        tokenizer = self.tokenizer
        train_df, test_df = load_data(self.tokenizer)

        X_train = extract_finetuned_embeddings(train_df["text"].tolist(), fine_tuned_model, tokenizer, batch_size=ANNConfig.BATCH_SIZE)
        X_test = extract_finetuned_embeddings(test_df["text"].tolist(), fine_tuned_model, tokenizer, batch_size=ANNConfig.BATCH_SIZE)

        self.input_dim = X_train.shape[1]

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
            X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
        )

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    '''
        Loads the custom trainer class,
        preps the data for training
        creates the ANN model ready for training
    '''
    def load_trainer(self):
        self.__check_model_presence__()
        self.model = ANNClassifier(self.input_dim)
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.trainer = Training(model=self.model)
        self.trainer.prepare_data(X_train, y_train, X_test, y_test)
        print("Trainer Initialized")


    def train(self):
        self.__check_model_presence__()

        self.trainer.train()
        print("Training Complete. Call 'evaluate()' for results")

    def evaluate(self):
        self.trainer.evaluate()


    def __check_model_presence__(self):
        if not self.ft_model or not self.tokenizer:
            raise ValueError("FineTuned Model not Loaded. Please Check Logs for details or try calling 'load_finetuned_model()'")
            #TODO: Add Logging