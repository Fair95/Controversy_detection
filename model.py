import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, BertForSequenceClassification, BertModel
from transformers import RobertaModel, DebertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoModel
from constants import n_topics, n_topics, id2label, label2id

## BERT model
class MyBertModel(nn.Module):
    def __init__(self):
        super(MyBertModel, self).__init__()

    def construct_bert(self, bert_variant, dropout_rate=0.2):
        kwargs = self.construct_model_config(dropout_rate)
        if bert_variant == 'roberta-base':
            bert = RobertaForSequenceClassification.from_pretrained('roberta-base', **kwargs)
        elif bert_variant == 'ProsusAI/finbert':
            finbert_without_classifier_head = AutoModel.from_pretrained("ProsusAI/finbert")
            finbert_without_classifier_head.save_pretrained("finbert_no_head")
            bert = AutoModelForSequenceClassification.from_pretrained("finbert_no_head", **kwargs)
        elif bert_variant == 'bert-base-cased':
            bert = BertForSequenceClassification.from_pretrained('bert-base-cased', **kwargs)
        elif bert_variant == 'microsoft/deberta-base':
            bert = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', **kwargs)
        else:
            raise NotImplementedError
        return bert

    def construct_bert_2(self, bert_variant, dropout_rate=0.2):
        if bert_variant == 'roberta-base':
            bert = RobertaModel.from_pretrained('roberta-base', add_pooling_layer = False, output_hidden_states=True)
        elif bert_variant == 'ProsusAI/finbert':
            bert = AutoModel.from_pretrained("ProsusAI/finbert", add_pooling_layer = False, output_hidden_states=True)
        elif bert_variant == 'bert-base-cased':
            bert = BertModel.from_pretrained('bert-base-cased', add_pooling_layer = False, output_hidden_states=True)
        elif bert_variant == 'microsoft/deberta-base':
            bert = AutoModel.from_pretrained('microsoft/deberta-base', add_pooling_layer = False, output_hidden_states=True)
        elif bert_variant == 'sentence-transformers/all-distilroberta-v1':
            bert = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1', add_pooling_layer=False, output_hidden_states=True)
        else:
            raise NotImplementedError
        return bert

    def construct_model_config(self, dropout_rate):
        kwargs = {
            'problem_type':"multi_label_classification", 
            'num_labels':n_topics,
            'id2label':id2label,
            'label2id':label2id,
            'classifier_dropout':dropout_rate 
        }

        return kwargs

    def forward(self, inputs):
        super().forward()

class MyBertModel_1(MyBertModel):
    def __init__(self, bert_variant):
        super(MyBertModel, self).__init__()
        self.name = bert_variant
        self.bert = self.construct_bert(bert_variant, dropout_rate=0.2)

    def forward(self, inputs, input_embed=False):
        if input_embed:
            inputs_embeds = inputs[0]
            attention_mask = inputs[1]
            bert_out = self.bert(attention_mask=attention_mask, inputs_embeds = inputs_embeds)
        else:
            input_ids = inputs[0]
            attention_mask = inputs[1]
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = bert_out.logits
        return out

class MyBertModel_2(MyBertModel):
    def __init__(self, bert_variant):
        super(MyBertModel_2, self).__init__()
        self.name = bert_variant + '_head'
        self.bert = self.construct_bert_2(bert_variant, dropout_rate=0.2)
        self.down = DownSampler(num_layers=3, dropout_rate=0.2, in_features=4 * 768, out_features=n_topics)

    def forward(self, inputs, input_embed=False):
        if input_embed:
            inputs_embeds = inputs[0]
            attention_mask = inputs[1]
            bert_out = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        else:
            input_ids = inputs[0]
            attention_mask = inputs[1]
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[1]
        pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        pooled_output = pooled_output[:, 0, :]

        out = self.down(pooled_output)

        return out

class DownSampler(nn.Module):
    def __init__(self, num_layers, dropout_rate, in_features, out_features):
        super(DownSampler, self).__init__()

        self.down = nn.ModuleList()
        for i in range(0, num_layers-1):
            self.down.append(nn.Dropout(p=dropout_rate))
            linear_layer = nn.Linear(in_features//(2**i), in_features//(2**(i+1)))
            self.down.append(linear_layer)
            self.down.append(nn.ReLU())
        self.down.append(nn.Dropout(p=dropout_rate))
        self.down.append(nn.Linear(in_features//(2**(i+1)), out_features))

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return x


## RNN model

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim

        self.concat_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, rnn_outputs, final_hidden_state):
        batch_size, seq_len, _ = rnn_outputs.shape
        attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        attn_weights = torch.bmm(attn_weights, final_hidden_state)

        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1).transpose(1,2)))

        return attn_hidden, attn_weights

class MyLSTMModel(nn.Module):
    def __init__(self, config, embedding_dim, hidden_dim, vocab_size, n_classes=n_topics):
        super(MyLSTMModel, self).__init__()
        self.name = 'lstm'
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_attention = self.config['use_attention']
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.num_directions = 2 if self.config['bidirectional'] == True else 1

        self.rnn = nn.LSTM(input_size=self.embedding_dim,
                       hidden_size=self.hidden_dim,
                       num_layers=self.config['num_layers'],
                       bidirectional=self.config['bidirectional'],
                       dropout=self.config['dropout_rate'],
                       batch_first=False)

        self.down = DownSampler(3, self.config['dropout_rate'], self.hidden_dim * self.num_directions, n_classes)

        if self.use_attention:
            self.attn = Attention(self.hidden_dim * self.num_directions)


    def forward(self, inputs):
        inputs = inputs[0]
        batch_size = inputs.shape[0]
        embed = self.word_embeddings(inputs).transpose(0, 1)

        rnn_output, self.hidden = self.rnn(embed)

        final_state = self.hidden[0].view(self.config['num_layers'], self.num_directions, batch_size, self.hidden_dim)[-1]

        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1) 
        attn_weights = None
        if self.use_attention:
            rnn_output = rnn_output.permute(1, 0, 2) 
            attn_out, self.attn_weights = self.attn(rnn_output, final_hidden_state.unsqueeze(2))
        else:
            attn_out = final_hidden_state

        out = self.down(attn_out.squeeze(1))

        return out

if __name__ == '__main__':
    bert_variant = 'ProsusAI/finbert'
    mybert = MyBertModel_2(bert_variant)
    # print(mybert.bert)

    from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
    from torch.utils.data import DataLoader
    from utils import get_df
    from data import dataset
    import torch

    path = 'cleaned_2.csv'
    tk = AutoTokenizer.from_pretrained(bert_variant)
    df_train, df_val, df_test = get_df(path)

    train_data = dataset(df_train,tk)
    train_dataloader = DataLoader(dataset = train_data, batch_size=1, shuffle=True)

    inputs = next(iter(train_dataloader))
    print(mybert(inputs))

