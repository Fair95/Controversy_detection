import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, configure_interpretable_embedding_layer, IntegratedGradients, LayerConductance

class BertEmbeddingView:
    def __init__(self, model, df, tk, device, topic_list):
        self.model = model
        self.df = df
        self.tk = tk
        self.device = device
        self.topic_list = topic_list

        self.model.to(self.device)
        self.model.eval()

        self.ref_token_id = self.tk.pad_token_id
        self.sep_token_id = self.tk.sep_token_id
        self.cls_token_id = self.tk.cls_token_id

        self.vis_data_records = []

    def reset(self):
        self.vis_data_records = []

    def get_input_label(self, index, target_class):
        sent = self.df.paragraph[index]
        label = self.df[self.topic_list[target_class]][index]
        return sent, label

    def construct_input_ref_pair(self, sent, max_len=510):
        token_ids = self.tk.encode(sent, add_special_tokens=False)[:max_len]

        # construct input token ids
        input_ids = [self.cls_token_id] + token_ids + [self.ref_token_id] * (max_len-len(token_ids)) + [self.sep_token_id]

        # construct reference token ids 
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * (max_len) + [self.sep_token_id]

        return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device), len(token_ids)

    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def construct_whole_bert_embeddings(self, input_ids, ref_input_ids):
        input_embeddings = self.model.bert.embeddings(input_ids)
        ref_input_embeddings = self.model.bert.embeddings(ref_input_ids)
        
        return input_embeddings, ref_input_embeddings

    def predict(self, input_ids, attention_mask=None):
        inputs = (input_ids, attention_mask)
        output = self.model(inputs)
        return output

    def predict_with_embed(self, inputs_embeds, attention_mask=None):
        inputs = (inputs_embeds, attention_mask)
        output = self.model(inputs, input_embed=True)
        return output

    def sigmoid_forward_func(self, input_ids, attention_mask=None):
        pred = self.predict(input_ids,attention_mask=attention_mask)
        return torch.sigmoid(pred).squeeze(0)

    def interpret_sentence(self, index, target_class):
        sent, label = self.get_input_label(index, target_class)
        input_ids, ref_input_ids, sep_id = self.construct_input_ref_pair(sent)
        attention_mask = self.construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        self.all_tokens = self.tk.convert_ids_to_tokens(indices)

        self.model.zero_grad()
        
        # predict
        pred = self.sigmoid_forward_func(input_ids, attention_mask)[target_class].item()
        pred_ind = round(pred)

        lig = LayerIntegratedGradients(self.predict, self.model.bert.embeddings)
        
        attributions, delta = lig.attribute(inputs=input_ids,
                                          baselines=ref_input_ids,
                                          additional_forward_args=attention_mask,
                                          return_convergence_delta=True,
                                          target=target_class,
                                          internal_batch_size=5)

        self.add_attributions_to_visualizer(attributions,
                                       self.all_tokens, 
                                       pred, 
                                       pred_ind, 
                                       label, 
                                       delta, 
                                       target_class)

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        return attributions

    def add_attributions_to_visualizer(self, attributions, text, pred, pred_ind, label, delta, target_class):
        attributions = self.summarize_attributions(attributions)
        pred_string = self.topic_list[target_class] if pred else 'Not '+ self.topic_list[target_class]
        label_string = self.topic_list[target_class] if label else 'Not '+ self.topic_list[target_class]
        # storing couple samples in an array for visualization purposes
        self.vis_data_records.append(visualization.VisualizationDataRecord(
                                attributions,
                                pred,
                                pred_string,
                                label_string,
                                self.topic_list[target_class],
                                attributions.sum(),
                                text,
                                delta))

    def save_sent_visualization(self, path):
        html = visualization.visualize_text(self.vis_data_records)
        with open(path,'wb') as f:   # Use some reasonable temp name
            f.write(html.data.encode("UTF-8"))

    def interpret_layers(self, index, target_class, token_to_explain=None):
        sent, label = self.get_input_label(index, target_class)
        input_ids, ref_input_ids, sep_id = self.construct_input_ref_pair(sent)
        attention_mask = self.construct_attention_mask(input_ids)
        input_embeddings, ref_input_embeddings = self.construct_whole_bert_embeddings(input_ids, ref_input_ids)
        layer_attrs = []
        layer_attrs_token = [] if token_to_explain else None

        indices = input_ids[0].detach().tolist()
        self.all_tokens = self.tk.convert_ids_to_tokens(indices)
        self.all_tokens = [x for x in self.all_tokens if x != '[PAD]']
        self.model.zero_grad()
        
        for i in range(self.model.bert.config.num_hidden_layers):
            lc = LayerConductance(self.predict_with_embed, self.model.bert.encoder.layer[i])
            layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, 
                                              additional_forward_args=attention_mask, target=target_class, 
                                              internal_batch_size=5 )
            layer_attrs.append(self.summarize_attributions(layer_attributions))
            if token_to_explain:
                # storing attributions of the token id that we would like to examine in more detail in token_to_explain
                layer_attrs_token.append(layer_attributions[0,token_to_explain,:].cpu().detach().tolist())
        return layer_attrs, layer_attrs_token

    def save_layer_visualization(self, layer_attrs, path, percentile=0.95):
        attrs_arr = np.array(layer_attrs)
        token_attr_avg = attrs_arr.mean(axis=0)
        top_quantile_tokens_index = token_attr_avg>np.quantile(token_attr_avg, percentile)
        

        fig, ax = plt.subplots(figsize=(15,5))
        xticklabels=[x for i, x in enumerate(self.all_tokens) if top_quantile_tokens_index[i]]
        yticklabels=list(range(1, self.model.bert.config.num_hidden_layers+1))
        ax = sns.heatmap(attrs_arr[:, top_quantile_tokens_index], xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.tight_layout()
        plt.savefig(path)

    def save_layer_token_visualization(self, layer_attrs_token, token_to_explain, path='emb_vis/layer_token_vis.png'):
        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.boxplot(data=layer_attrs_token)
        plt.title(self.all_tokens[token_to_explain])
        plt.xlabel('Layers')
        plt.ylabel('Attribution')
        plt.tight_layout()
        plt.savefig(path)

    def run_sentence_interpret(self, trainer, n_topics):
        for i in range(n_topics):
            print('Intepret sentences for topic {}'.format(self.topic_list[i]))
            target_samples_index = self.df[self.df[self.topic_list[i]]].index
            for index in target_samples_index:
                sent = self.df['paragraph'][index]
                data_encoded = self.tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
                input_ids = data_encoded['input_ids']
                attention_mask = data_encoded['attention_mask']
                inputs = (input_ids, attention_mask)
                pred = trainer.inference(inputs)[i]
                if pred == 0:
                    print(index)
                    self.interpret_sentence(index=index, target_class=i)
                    path = 'emb_vis/captum_bert_vis_class_{}_wrong.htm'.format(i)
                    self.save_sent_visualization(path)
                    self.reset()
                    break
            for index in target_samples_index:
                sent = self.df['paragraph'][index]
                data_encoded = self.tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
                input_ids = data_encoded['input_ids']
                attention_mask = data_encoded['attention_mask']
                inputs = (input_ids, attention_mask)
                pred = trainer.inference(inputs)[i]
                if pred == 1:
                    print(index)
                    self.interpret_sentence(index=index, target_class=i)
                    path = 'emb_vis/captum_bert_vis_class_{}_correct.htm'.format(i)
                    self.save_sent_visualization(path)
                    self.reset()
                    break               
        target_samples_index = self.df[self.df.apply(lambda x: sum(eval(x['label'])) > 1,axis=1)].index
        row = self.df.loc[target_samples_index[0]]
        non_zero_class = np.nonzero(eval(row.label))
        for i in non_zero_class[0]:
            self.interpret_sentence(index=target_samples_index[0], target_class=int(i))
        path = 'emb_vis/captum_bert_vis_multi.htm'
        self.save_sent_visualization(path)

    def run_layer_interpret(self, trainer, example_index=0, token_to_explain=10):
        target_samples_index = self.df[self.df.apply(lambda x: sum(eval(x['label'])) > 1,axis=1)].index
        row = self.df.loc[target_samples_index[example_index]]
        non_zero_class = np.nonzero(eval(row.label))
        for i in non_zero_class[0]:
            layer_attrs, layer_attrs_dist = bert_vis.interpret_layers(target_samples_index[example_index], int(i), token_to_explain)
            self.save_layer_visualization(layer_attrs, path='emb_vis/layer_vis_class_{}.png'.format(int(i)), percentile=0.98)
            self.save_layer_token_visualization(layer_attrs_dist, token_to_explain, path='emb_vis/layer_token_vis_class_{}.png')
        path = 'emb_vis/captum_bert_vis_multi.htm'
        self.save_sent_visualization(path)

def plot_lstm_attn(model, tk, df, index, device, topic_list):
    sent = df.paragraph.iloc[index]
    label = '_'.join([topic_list[i] for i, x in enumerate((eval(df.label.iloc[index]))) if x == 1])
    data_encoded = tk(sent, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    all_tokens = tk.convert_ids_to_tokens(data_encoded['input_ids'][0])
    all_tokens = [x+'_'+str(i) for i, x in enumerate(all_tokens)]
    inputs = (data_encoded['input_ids'], data_encoded['attention_mask'])
    inputs = [x.to(device) for x in inputs]
    model = model.to(device)
    _ = model(inputs)
    attn_weights = model.attn_weights.squeeze(2)
    attn_weights = F.normalize(attn_weights, p=2, dim=1).detach().cpu().numpy()[0]
    top_index = np.argpartition(attn_weights, -10)[-10:]
    plt.barh(np.array(all_tokens)[top_index], attn_weights[top_index])
    plt.title(label)
    plt.tight_layout()
    plt.savefig('emb_vis/lstm_attn_weights.png')

if __name__ == '__main__':
    import pandas as pd
    from utils import get_df
    df_train, df_val, df_test = get_df('cleaned_2.csv')
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    from transformers import AutoTokenizer
    from trainer import Trainer
    from model import MyBertModel_1, MyBertModel_2, MyLSTMModel
    from constants import n_topics, topic_list
    plt.rcParams.update({'font.size': 17})
    bert_variant = 'ProsusAI/finbert'
    trainer = Trainer(MyBertModel_2(bert_variant))
    trainer.from_checkpoint(model_path='models/ProsusAI_finbert_head_3e-06_10_512_False_None_saved_model.pt')

    tk = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    bert_vis = BertEmbeddingView(trainer.model, df_test, tk, device, topic_list)
    bert_vis.run_sentence_interpret(trainer, n_topics)
    bert_vis.run_layer_interpret(trainer, example_index=0, token_to_explain=3667)
    
    # LSTM attn_weights
    import torch.nn.functional as F
    from my_tokenizers import construct_tokenizer
    tk = construct_tokenizer(df_train.paragraph, 100)
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
    trainer = Trainer(MyLSTMModel(config=lstm_config, embedding_dim=768,
                                    hidden_dim=256,
                                    vocab_size=tk.vocab_size))

    trainer.from_checkpoint(model_path='models/lstm_0.0001_256_True_True_0.2_2saved_model.pt')
    plot_lstm_attn(trainer.model, tk, df_test, 15, device, topic_list)

    # _ = visualization.visualize_text(vis_data_records_ig)

