import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import random

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
from termcolor import colored

from utils import save, load

class Preprocessor(object):
    """
    Preprocessor
    """
    def __init__(self):
        # Define spacy pipeline
        self.nlp = spacy.load('en_core_web_md')
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x, min_df = 2, max_df = 0.5, ngram_range = (1, 2), lowercase=False, max_features=1000)

    def spacy_preprocess(self, texts, save_path, batch_size=2000, force_rebuild=False):
        """
        Function to preprocess using spacy tokenizer AND save the resulted tokens
        texts: List of texts
        save_path: path to save
        force_rebuild: bool, whether to force the re-preprocessing of the df

        Return: List of tokens
        """
        if os.path.exists(save_path) and not force_rebuild:
            print('Loading preprocessed tokens...')
            out_tokens = load(save_path)
            print('Done!\nPreprocessed token loaded at {}'.format(save_path))
        else:
            print('Preprocessing using spacy pipeline...')
            if batch_size:
                out_tokens = self.batch_process(texts, batch_size)
            else:
                print('Whole Processing, Whole size = {}'.format(len(texts)))
                docs = self.nlp.pipe(texts)
                out_tokens = [self.spacy_tokenizer(doc) for doc in docs]
            save(out_tokens, save_path)
            print('Done!\nPreprocessed token saved at {}'.format(save_path))
        return out_tokens

    def batch_process(self, texts, batch_size):
        print('Batch Processing, Batch size = {}'.format(batch_size))
        out_tokens = []
        len_docs = len(texts)
        n_loops = (len_docs//batch_size)+1
        for i in range(n_loops):
            print('[Batch {}/{}]...'.format(i, n_loops))
            docs = self.nlp.pipe(texts[i*batch_size: (i+1)*batch_size])
            tmp_out_tokens = [self.spacy_tokenizer(doc) for doc in docs]
            out_tokens.extend(tmp_out_tokens)
        return out_tokens

    @staticmethod 
    def spacy_tokenizer(doc):
        """
        A spacy tokenizer to do preprocessing for Tranditional ML classifier
        """
        out_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue # Remove stop words
            elif token.like_email:
                out_tokens.append('<email>') # Replayce email 
            elif token.like_url:
                out_tokens.append('<url>') # Replace URL
            elif token.like_num:
                out_tokens.append('<num>') # Replace numbers
            elif token.lemma_ != "-PRON-": # Lemmatisation if not proper noun
                out_tokens.append(token.lemma_.lower().strip()) 
            else:
                out_tokens.append(token.lower_) # Lower case
        return out_tokens

    def get_tfidf_vectors(self, df, train, load_path='spacy_preprocessed', batch_size=2000, force_rebuild=False):
        """
        Function to build TF-IDF matrix
        df: dataframe to process
        train: bool, if true fit and transform, if false, only transform
        load_path: if present, load tokens from file (No preprocessing of df required)
        force_rebuild: bool, whether to force the re-preprocessing of the df

        Return: TF-IDF matrix, size (#samples, #tokens)
        """
        path = load_path+'_train.pkl' if train else load_path+'_val.pkl'
        out_tokens = self.spacy_preprocess(df.paragraph, save_path=path, batch_size=batch_size, force_rebuild=force_rebuild)
        
        if train:
            features = self.tfidf.fit_transform(out_tokens)
        else:
            features = self.tfidf.transform(out_tokens)
        
        labels = np.array([eval(l) for l in df.label])
        return features, labels


def preprocess(nlp, paragraph):
    tokens_list = []
    sent_list = []
    num = 0
    email = 0
    url = 0
    bracket = 0
    quote = 0
    currency = 0
    oov = 0
    for doc in nlp.pipe(paragraph):
        tokens = defaultdict(list)
        sent_list.append(doc)
        for token in doc:
            tokens['tokens'].append(token)
            if token.like_num:
                tokens['num'].append(token)
                num += 1
            if token.like_email:
                tokens['email'].append(token)
                email += 1
            if token.like_url:
                url +=1
                tokens['url'].append(token)
            if token.is_bracket:
                bracket += 1
            if token.is_quote:
                quote += 1
            if token.is_currency:
                currency += 1
                tokens['currency'].append(token)
            if token.is_oov:
                oov += 1
                tokens['oov'].append(token)
            # if len(tokens) > 3:
            #     tokens.append(doc) 
            # print(token.text, token.like_num, nlp.vocab.strings[token.text])
        # break
        tokens_list.append(tokens)
    return tokens_list, (num,email,url,bracket,quote,currency,oov), sent_list








