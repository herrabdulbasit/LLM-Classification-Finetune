from data_prep import DataPrep

# Util functions !!!
def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    return [len(t) for t in input_ids]

def load_data(tokenizer):
    from sklearn.model_selection import train_test_split

    train = DataPrep("dataset/train.csv").perform()
    train.loc[:, 'token_count'] = get_token_lengths(train['text'], tokenizer)
    train = train[train["token_count"]<= 1024]
    print('Total train samples: ', len(train))
    train, eval_df = train_test_split(train, test_size=0.1, random_state=42)

    return train, eval_df