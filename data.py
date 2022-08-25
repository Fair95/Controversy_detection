import torch
from torch.utils.data import Dataset

## This file contains Dataset class
class dataset(Dataset):
    def __init__(self, df, tokenizer, test=False):
        self.df = df
        self.tk = tokenizer
        self.test = test
        self.data_encoded = [self.tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for sent in self.df['paragraph']]
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        data_encoded = self.data_encoded[index]
        if self.test:
            return data_encoded['input_ids'].squeeze(), data_encoded['attention_mask'].squeeze()
        else:
            label = torch.FloatTensor([eval(self.df['label'].iloc[index])]).squeeze()
            return data_encoded['input_ids'].squeeze(), data_encoded['attention_mask'].squeeze(), label

    def get_sample_weights(self, scaling=1):
        weights = self.df['label'].value_counts().min() / self.df['label'].value_counts()[self.df['label']]
        weights = weights.to_list()
        weights = [w*scaling if w!=1 else w for w in weights]
        return weights


if __name__=='__main__':
    from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
    from torch.utils.data import DataLoader
    from utils import get_df

    path = 'cleaned_2.csv'
    tk = AutoTokenizer.from_pretrained("roberta-base")
    df_train, df_val, df_test = get_df(path)


    train_data = dataset(df_train,tk)
    val_data = dataset(df_val,tk)
    test_data = dataset(df_test, tk, test=True)
    train_dataloader = DataLoader(dataset = train_data, batch_size=8, shuffle=True)
    # val_dataloader = DataLoader(dataset = val_data, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(dataset = test_data, batch_size=1, shuffle=False)

    a,b = next(iter(test_dataloader))
    print(a.shape, b.shape)