from upsetplot import UpSet
from upsetplot import from_memberships, from_indicators
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd

from data_analysis import Preprocessor

def plot_distribution(labels, data):
    data_by_label = from_indicators(labels, data=data)
    distribution_plot = UpSet(data_by_label, show_counts=True, min_subset_size=100)
    return distribution_plot


def plot_word_cloud(text, stopwords):
    # Create and generate a word cloud image
    wordcloud = WordCloud(max_font_size=50, stopwords=stopwords, max_words=100, background_color="white").generate(text)

    return wordcloud

if __name__ == '__main__':

    from constants import topic_code, topic_list, n_topics
    
    # Read data
    news_data = 'cleaned_2.csv'
    news = pd.read_csv(news_data)

    # Plot distribution
    has_topic = [] # Find all labels
    for topic in topic_list:
        if topic in news.columns:
            has_topic.append(topic)
    distribution_plot = plot_distribution(has_topic, news)
    distribution_plot.plot()

    # Plot word cloud
    pre = Preprocessor()
    news = pd.read_csv('cleaned_2.csv')
    for topic in topic_list:
        print('[{}]'.format(topic))
        news_by_topic = news[news[topic]==True]
        # Process cleaned text using spacy
        out_tokens = pre.spacy_preprocess(news_by_topic['paragraph'], save_path='{}_tokens'.format(topic))
        # Join all words for the topic
        text = " ".join(" ".join(out) for out in out_tokens)
        print(len(text))
        stopwords = ['num', 'email', '<num>', '<email>','thomson', 'reuters', 'year',
                     'month','storycontent','Refinitiv', ]
        word_cloud = plot_word_cloud(text, stopwords)
        # Display the generated image:
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('{}_wordcloud.png'.format(topic))
