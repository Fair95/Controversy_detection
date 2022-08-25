import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
from data import dataset
from model import MyBertModel_1, MyBertModel_2, MyLSTMModel
from trainer import Trainer
from utils import get_df
from data_analysis import Preprocessor
from my_tokenizers import construct_tokenizer
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(1)

# Check if using cuda
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# Useful settings (hyperparameters)
config = {
    'preprocess': False,
    'use_layerwise_learning_rate': True,
    'lr': 1e-5,
    'epochs': 10,
    'gradient_accumulate_steps': 2,
    'resample_scale': None,
    'input_max_length': 512,
    'batch_size': 5
}

lstm_config = {
    'preprocess': False,
    'use_layerwise_learning_rate': False,
    'lr': 1e-4,
    'epochs': 20,
    'gradient_accumulate_steps': 1,
    'resample_scale': 1,
    'input_max_length': 256,
    'batch_size': 20,
    'use_attention': True,
    'bidirectional': True,
    'dropout_rate': 0.2,
    'num_layers': 2,

}

# Load data
path = 'cleaned_2.csv'
df_train, df_val, df_test = get_df(path)

# Preprocessing
bert_variant = None

# Define tokenizer/Bert variant
# bert_variant = "roberta-base"
# tk = AutoTokenizer.from_pretrained(bert_variant)
# bert_variant = RobertaForSequenceClassification.from_pretrained(model_name, **kwargs)


# Finbert

if bert_variant == 'ProsusAI/finbert':
    tk = AutoTokenizer.from_pretrained(bert_variant)

# Bert
elif bert_variant == "bert-base-cased":
    tk = AutoTokenizer.from_pretrained(bert_variant)

# Deberta
elif bert_variant == "microsoft/deberta-base":
    tk = AutoTokenizer.from_pretrained(bert_variant)

elif bert_variant == 'sentence-transformers/all-distilroberta-v1':
    tk = AutoTokenizer.from_pretrained(bert_variant)

elif bert_variant == None:
    tk = construct_tokenizer(df_train.paragraph, 100)

else:
    raise NotImplementedError
    
# Prepare dataset
train_data = dataset(df_train, tk)
val_data = dataset(df_val, tk)
test_data = dataset(df_test, tk, test=True)

# Rebalance data if necessary
if config['resample_scale'] is not None:
    train_sample_weights = train_data.get_sample_weights(scaling=config['resample_scale'])
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_data), replacement=True)
else:
    train_sampler = RandomSampler(train_data, replacement=False)

# Prepare dataloader
train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'], sampler=train_sampler)
val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'], shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

# Define our Trainer class
if bert_variant is not None:
    trainer = Trainer(MyBertModel_2(bert_variant), config, train_dataloader, val_dataloader)
else:
    trainer = Trainer(MyLSTMModel(config=lstm_config, embedding_dim=768,
                                    hidden_dim=256,
                                    vocab_size=tk.vocab_size),
                        lstm_config, train_dataloader, val_dataloader)

# -- Start Training -- #
trainer.train(val_freq=1)


# If load from pretrained
trainer.from_checkpoint(model_path=trainer.save_path)

# Do test inference
# trainer.metric.reset()
# for i, data in enumerate(test_dataloader):
#     y_pred = trainer.inference(data)


