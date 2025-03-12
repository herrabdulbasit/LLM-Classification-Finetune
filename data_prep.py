import pandas as pd
class DataPrep:
    def __init__(self, file_path):
        self._file_path_ = file_path

    def perform(self):
        df = self.__load_data()
        data = []
        for _, row in df.iterrows():
            data.append({
                "prompt": row["prompt"],
                "response_A": row["response_a"],
                "response_B": row["response_b"],
                "label": self.__get_label(row)
            })

        cleaned = pd.DataFrame(data)
        cleaned["prompt"] = cleaned["prompt"].str.replace(r"[\[\]]", "", regex=True).replace(r'^"|"$', '', regex=True)
        cleaned["response_A"] = cleaned["response_A"].str.replace(r"[\[\]]", "", regex=True).replace(r'^"|"$', '', regex=True)
        cleaned["response_B"] = cleaned["response_B"].str.replace(r"[\[\]]", "", regex=True).replace(r'^"|"$', '', regex=True)
        cleaned.to_csv("dataset/cleaned_train.csv", index=False)

        return cleaned

    def __load_data(self):
        return pd.read_csv(self._file_path_)

    def __get_label(self, row):
        '''
        Lets decide that:
        0: Winner Model a
        1: Winner Model b
        2: Tie
        '''
        if row["winner_model_a"] == 1:
            return 0
        elif row["winner_model_b"] == 1:
            return 1
        else:
            return 2 


def main():
    DataPrep("dataset/train.csv").perform()


if __name__ == "__main__":
    main()