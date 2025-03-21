import pandas as pd
import numpy as np

'''
Performs Data Preparations by removing nulls, special characters
Then also, labels the data
'''
class DataPrep:
    def __init__(self, file_path):
        self._file_path_ = file_path

    def perform(self):
        df = self.__load_data()

        def process(input_str):
            stripped_str = input_str.strip('[]')
            sentences = [s.strip('"') for s in stripped_str.split('","')]
            return  ' '.join(sentences)

        df.loc[:, 'prompt'] = df['prompt'].apply(process)
        df.loc[:, 'response_a'] = df['response_a'].apply(process)
        df.loc[:, 'response_b'] = df['response_b'].apply(process)

        indexes = df[(df.response_a == 'null') & (df.response_b == 'null')].index
        df.drop(indexes, inplace=True)
        df.reset_index(inplace=True, drop=True)

        df["text"] = (
            "[USER PROMPT]: " + df["prompt"] + "\n\n"
            "[MODEL A]: " + df["response_a"] + "\n\n"
            "[MODEL B]: " + df["response_b"] + "\n\n"
            "Which response is better? (0) Model A, (1) Model B, (2) Tie"
        )

        df.loc[:, 'label'] = np.argmax(df[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)

        return df


    def __load_data(self):
        return pd.read_csv(self._file_path_)
