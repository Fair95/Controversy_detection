import pandas as pd
from bs4 import BeautifulSoup
import os
import spacy
import re
import matplotlib.pyplot as plt
import numpy as np
from constants import topic_code, topic_list, n_topics

def html2content(story):
    def class_filter(css_class):
        useless_info = ['tr-by', 'tr-contactinfo', 'tr-signoff', 'tr-copyright',
                        'tr-slugline', 'line-break', 'tr-advisory']
        return css_class not in useless_info
    soup = BeautifulSoup(story,"lxml").find_all('p', class_=class_filter)
    res = ''
    for data in soup:
        res += (data.get_text()+'\n')

    return clean_html(res)

def clean_html(string):
    res = re.sub('\(Reporting by .*', '', string)
    res = re.sub('\(c\) Copyright Thomson Reuters 2022. Click For Restrictions - https://agency.reuters.com/en/copyright.html','', res)
    res = re.sub('\(\(Reuters Investor Briefs .*', '', res)
    res = re.sub('\(c\) Copyright Refinitiv .*', '', res)
    return res

def preprocess(news):
    for topic in topic_list:
        news[topic] = news[topic].fillna(0).astype(bool) # for upset plot

    # Exclude those not found
    news = news[news['story']!= 'Not found'] 

    # Generate labels
    news.loc[:, 'label'] = news.apply(lambda x: [int(x[tag]) for tag in topic_list], axis=1) 
    return news

def postprocess(news, max_len, min_len):
    # Exclude news that is too short or too long
    story_len = news['paragraph'].apply(len)
    news = news[np.logical_and((story_len <= max_len), (story_len >= min_len))] 
    return news

def run_cleaning(raw_data_path, max_len, min_len, save_path='cleaned.csv'):
    print('[Cleaning the raw data]...')
    if os.path.exists(save_path):
        print('Cleaned file already exists, skipping...')
        return

    news = pd.read_csv(raw_data_path)
    # Preprocess
    news = preprocess(news)
    # Clean html
    news.loc[:, 'paragraph'] = news['story'].apply(html2content)
    # Postprocess
    news = postprocess(news, max_len, min_len)
    # Save
    news.to_csv(save_path, index=False)

    return
if __name__=='__main__':
    news_data = 'second_test.csv'
    run_cleaning(news_data, max_len=5000, min_len=200, save_path='cleaned_2.csv')

    