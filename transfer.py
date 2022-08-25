import pandas as pd
import random
import glob
import re   
from tqdm import tqdm 
from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer, AutoModelForPreTraining, AutoModelForMaskedLM, AutoModelForNextSentencePrediction
from data import dataset
from model import MyBertModel_1, MyBertModel_2, MyLSTMModel
from trainer import Trainer
from utils import get_df
from data_analysis import Preprocessor
from my_tokenizers import construct_tokenizer
from constants import n_topics, topic_list

class PreTrainHeadForLSTM(MyLSTMModel):
    def __init__(self, config, embedding_dim, hidden_dim, vocab_size, n_classes):
        super(PreTrainHeadForLSTM, self).__init__(config, embedding_dim, hidden_dim, vocab_size, n_classes)
    def forward(self, inputs):
        inputs = inputs
        batch_size = inputs.shape
        embed = self.word_embeddings(inputs).transpose(0, 1)
        rnn_output, self.hidden = self.rnn(embed)
        final_hidden_state = rnn_output.permute(1, 2, 0)
        if self.use_attention:
            rnn_output = rnn_output.permute(1, 0, 2) 
            attn_out, self.attn_weights = self.attn(rnn_output, final_hidden_state)
        else:
            attn_out = final_hidden_state

        out = self.down(attn_out)
        return out 

class TransferDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def get_transfer_data(path):
    def get_label(l):
        label = [0] * n_topics
        for c in l:
            label[c] = 1
        return str(label)
    def get_indicator(label):
        indicator = []
        for i, c in enumerate(label):
            indicator.append(True) if c else indicator.append(False)
        return pd.Series(indicator, index=topic_list)
    df = pd.read_csv(path)
    df = df.groupby('id').agg({'story':'first', 'label': lambda x: get_label(x)})
    df['paragraph'] = df.story
    indicator_df = df.apply(lambda row: get_indicator(eval(row.label)), axis='columns', result_type='expand')
    df = pd.concat([df, indicator_df], axis='columns')
    df.reset_index(inplace=True)
    return df

def get_contextual_words(tk):
    txt_files = glob.glob("keywords/*.txt")

    def read_first_line(file):
        with open(file, 'rt') as fd:
            first_line = fd.readline()
        return first_line

    output_strings = map(read_first_line, txt_files)
    keywords_list = list(output_strings)

    keywords_string = ' '.join(keywords_list)
    expr = r"[\'\"\(\)]|\bOR"
    keywords_clean = re.sub(expr, '', keywords_string)
    keywords_encoded = tk.encode(keywords_clean)
    keywords_encoded.remove(tk.cls_token_id)
    keywords_encoded.remove(tk.sep_token_id)

    contextual_words = list(set(keywords_encoded))
    print('# of contextual words:', len(contextual_words))
    return contextual_words

def construct_pretrain_dataset(tk, df, max_len, do_nsp, do_mlm, do_contextual):
    news_a = []
    news_b = []
    label = []
    if do_nsp:
        all_indices = list(df.index)
        for i in range(len(df)) :
            if len(all_indices) == 0:
                break
            first_index = random.choice(all_indices)
            all_indices.remove(first_index)
            same_label_set = df[df.label==df.label.iloc[first_index]]
            diff_label_set = df[df.label!=df.label.iloc[first_index]]
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                news_a.append(df.story.loc[first_index])
                same_label_indices = list(set(all_indices) & set(same_label_set.index))
                if len(same_label_indices) == 0:
                    second_index = first_index
                else:
                    second_index = random.choice(same_label_indices)
                    all_indices.remove(second_index)
                news_b.append(df.story.loc[second_index])
                label.append(0)
            else:
                diff_label_indices = list(set(all_indices) & set(diff_label_set.index))
                if len(diff_label_indices) == 0:
                    break
                news_a.append(df.story.loc[first_index])
                second_index = random.choice(diff_label_indices)
                all_indices.remove(second_index)
                news_b.append(df.story.loc[second_index])
                label.append(1)

        inputs = tk(news_a, news_b, return_tensors='pt',
                       max_length=max_len, truncation=True, padding='max_length')
        inputs['next_sentence_label'] = torch.LongTensor([label]).T
    else:
        inputs = tk(list(df.paragraph), return_tensors='pt',
                   max_length=max_len, truncation=True, padding='max_length')

    inputs['labels'] = -100 * torch.ones_like(inputs.input_ids)
    rand = torch.rand(inputs.input_ids.shape)
    if do_contextual:
        contextual_words = get_contextual_words(tk)
        ind = torch.ones(inputs.input_ids.shape, dtype=torch.bool)
        for id in contextual_words:
            ind = ind * (inputs.input_ids != id)

        mask_arr = ~ind * (rand<0.4)
    else:
        mask_arr = (rand < 0.15) * (inputs.input_ids != tk.cls_token_id) * \
          (inputs.input_ids != tk.sep_token_id) * (inputs.input_ids != tk.pad_token_id)
    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    print(selection)
    if do_mlm:
        for i in range(inputs.input_ids.shape[0]):
            inputs.labels[i, selection[i]] = inputs.input_ids[i, selection[i]]
            inputs.input_ids[i, selection[i]] = tk.mask_token_id
    print(inputs.input_ids, inputs.labels)
    return inputs

def freeze_layer(model, layers):
    for n, p in model.named_parameters():
        if any(nd in n for nd in layers):
            print('Notice: layer {} get freezed'.format(n))
            p.requires_grad=False

def pretrain_bert(model, epochs, lr, loader, device, do_nsp, do_mlm):
    optim = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0, # Default value
        num_training_steps=epochs * len(loader)
    )
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, bar_format="{bar}{l_bar}{r_bar}")
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # process
            if do_nsp and do_mlm:
                labels = batch['labels'].to(device)
                next_sentence_label = batch['next_sentence_label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                next_sentence_label=next_sentence_label,
                                labels=labels)
            elif not do_nsp and do_mlm:
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)
            else:
                labels = batch['next_sentence_label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            loop.set_description("Loss: ",loss.item())
            # update parameters
            optim.step()
            scheduler.step()
        print(loss.item())
    model.save_pretrained('after_transfer')


def pretrain_lstm(model, epochs, lr, loader, device):
    optim = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0, # Default value
        num_training_steps=epochs * len(loader) # Note the traing steps also adjust based on gas
    )
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, bar_format="{bar}{l_bar}{r_bar}")
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids)
            # extract loss
            loss = torch.nn.CrossEntropyLoss()(outputs.permute(0,2,1), labels)
            # calculate loss for every parameter that needs grad update
            loss.backward()
            loop.set_description("Loss: ",loss.item())
            # update parameters
            optim.step()
            scheduler.step()
        print(loss.item())
    torch.save(model.state_dict(), 'models/lstm_after_pretrain.pt')
def load_layer_parts(pretrain_path, model, device):
    pretrain_state_dict = torch.load(pretrain_path, map_location=device)
    pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if 'down' not in k}
    
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrain_state_dict) 
    model.load_state_dict(pretrain_state_dict, strict=False)

def evaluate_model(trainer, device, test_dataloader):
    trainer.metric.reset()
    trainer.model.to(device)
    for i, data in enumerate(test_dataloader):
        trainer.model.eval()
        with torch.no_grad():
            data = [x.to(device) for x in data]
            y_pred = trainer.model(data[:-1])
            y_true = data[-1]
            trainer.metric.update(y_pred, y_true)

def write_to_df(df, device):
    # Run bert
    bert_variant = "ProsusAI/finbert"
    tk = AutoTokenizer.from_pretrained(bert_variant)
    trainer_bert = Trainer(MyBertModel_2(bert_variant))
    trainer_bert.from_checkpoint(model_path='models/ProsusAI_finbert_head_1e-05_10_512_False_None_saved_model.pt')
    trainer_bert.model.bert = trainer_bert.model.bert.from_pretrained('after_transfer', add_pooling_layer = False, output_hidden_states=True)
    test_data = dataset(df, tk)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    evaluate_model(trainer_bert, device, test_dataloader)
    # Run lstm
    lstm_config = {
            'preprocess': False,
            'use_layerwise_learning_rate': False,
            'lr': 1e-4,
            'epochs': 1,
            'gradient_accumulate_steps': 1,
            'resample_scale': 1,
            'input_max_length': 256,
            'batch_size': 20,
            'use_attention': True,
            'bidirectional': True,
            'dropout_rate': 0.2,
            'num_layers': 2,

        }
    path = 'cleaned_2.csv'
    df_train, _, _ = get_df(path)
    tk_lstm = construct_tokenizer(df_train.paragraph, 100)
    trainer_lstm = Trainer(MyLSTMModel(config=lstm_config, embedding_dim=768,
                                        hidden_dim=256,
                                        vocab_size=tk_lstm.vocab_size))
    trainer_lstm.from_checkpoint('models/lstm_0.0001_256_True_True_0.2_2saved_model.pt')
    load_layer_parts('models/lstm_after_pretrain.pt', trainer_lstm.model, device)
    evaluate_model(trainer_lstm, device, test_dataloader)
    test_data = dataset(df, tk_lstm)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    bert_label = []
    lstm_label = []
    pseudo_label = []
    news = []
    for i in range(len(df)):
        bert_label.append([topic_list[j] for j, c in enumerate(trainer_bert.metric.y_pred[i]) if c == 1])
        lstm_label.append([topic_list[j] for j, c in enumerate(trainer_lstm.metric.y_pred[i]) if c == 1])
        pseudo_label.append([topic_list[j] for j, c in enumerate(eval(df.label.iloc[i])) if c == 1])
        news.append(df.paragraph.iloc[i])
    result_df = pd.DataFrame({'news_content': news, 'pseudo_label': pseudo_label, 'bert_label': bert_label, 'lstm_label':lstm_label})
    result_df.to_csv('transfer_result.csv')

if __name__ == '__main__':
    model_type = 'BERT'
    pretrain = True
    fine_tune = False
    do_nsp = True
    do_mlm = False
    do_contextual = False
    layer_to_freeze = [] #['layer.8.','layer.9.','layer.10.','layer.11.']
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
        'epochs': 1,
        'gradient_accumulate_steps': 2,
        'resample_scale': None,
        'input_max_length': 512,
        'batch_size': 5
    }


    # Load data
    path = 'cleaned_2.csv'
    df_train, df_val, df_test = get_df(path)

    # Preprocessing
    bert_variant = "ProsusAI/finbert"
    tk = AutoTokenizer.from_pretrained(bert_variant)

        
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


    path = 'twitter_data.csv'
    df = get_transfer_data(path)  

    if model_type == 'BERT':
        # BERT
        # If load from pretrained
        trainer = Trainer(MyBertModel_2(bert_variant), config, train_dataloader, val_dataloader)

        trainer.from_checkpoint(model_path='models/ProsusAI_finbert_head_1e-05_10_512_False_None_saved_model.pt')
        trainer.model.bert.save_pretrained('before_transfer')
        inputs = construct_pretrain_dataset(tk=tk, df=df, max_len=config['input_max_length'], do_nsp=do_nsp, do_mlm=do_mlm, do_contextual=do_contextual)
        transfer_dataset = TransferDataset(inputs)
        loader = torch.utils.data.DataLoader(transfer_dataset, batch_size=5, shuffle=True)
        if do_nsp and do_mlm:
            model = AutoModelForPreTraining.from_pretrained('before_transfer').to(device)
        elif not do_nsp and do_mlm:
            model = AutoModelForMaskedLM.from_pretrained('before_transfer').to(device)
        else:
            model = AutoModelForNextSentencePrediction.from_pretrained('before_transfer').to(device)
        freeze_layer(model, layer_to_freeze)
        if pretrain:
            pretrain_bert(model=model, epochs=2, lr=5e-6, loader=loader, device=device, do_nsp=do_nsp, do_mlm=do_mlm)
            trainer.model.bert = trainer.model.bert.from_pretrained('after_transfer', add_pooling_layer = False, output_hidden_states=True)
        test_data = dataset(df, tk)
        test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        evaluate_model(trainer, device, test_dataloader)
        print(trainer.metric)
        if fine_tune:
            # -- Start Training -- #
            trainer.train(val_freq=1)


    if model_type == 'LSTM':
        # LSTM
        lstm_config = {
            'preprocess': False,
            'use_layerwise_learning_rate': False,
            'lr': 1e-4,
            'epochs': 1,
            'gradient_accumulate_steps': 1,
            'resample_scale': 1,
            'input_max_length': 256,
            'batch_size': 20,
            'use_attention': True,
            'bidirectional': True,
            'dropout_rate': 0.2,
            'num_layers': 2,

        }
        tk_lstm = construct_tokenizer(df_train.paragraph, 100)
        inputs = construct_pretrain_dataset(tk=tk_lstm, df=df, max_len=lstm_config['input_max_length'], do_nsp=do_nsp, do_mlm=do_mlm, do_contextual=do_contextual)
        transfer_dataset = TransferDataset(inputs)
        loader = torch.utils.data.DataLoader(transfer_dataset, batch_size=5, shuffle=True)
        test_data = dataset(df, tk_lstm)
        test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        trainer_lstm = Trainer(MyLSTMModel(config=lstm_config, embedding_dim=768,
                                        hidden_dim=256,
                                        vocab_size=tk_lstm.vocab_size))
        trainer_lstm.from_checkpoint('models/lstm_0.0001_256_True_True_0.2_2saved_model.pt')
        if pretrain:
            model = PreTrainHeadForLSTM(config=lstm_config, embedding_dim=768,
                                hidden_dim=256,
                                vocab_size=tk_lstm.vocab_size,
                                n_classes=tk_lstm.vocab_size).to(device)
            load_layer_parts('models/lstm_0.0001_256_True_True_0.2_2saved_model.pt', model, device)
            pretrain_lstm(model=model, epochs=2, lr=5e-6, loader=loader, device=device)
            load_layer_parts('models/lstm_after_pretrain.pt', trainer_lstm.model, device)
        evaluate_model(trainer_lstm, device, test_dataloader)
        if fine_tune:
            # -- Start Training -- #
            trainer.train(val_freq=1)
        print(trainer_lstm.metric)

