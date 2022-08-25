import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

def get_df(path):
    df = pd.read_csv(path)
    df_train, df_val_test = train_test_split(df, train_size=0.80, random_state=1)
    df_val, df_test = train_test_split(df_val_test, test_size=0.50, random_state=1)

    df_train.to_csv('df_train.csv')
    df_val.to_csv('df_val.csv')
    df_test.to_csv('df_test.csv')
    return df_train, df_val, df_test

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
        ploter = pickle.load(f)
    return ploter

def evaluate(y_score, y_true):

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(y_score >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(classification_report(y_true, y_pred))
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



if __name__ == '__main__':
    # df_train, df_val, df_test, df_pcl, df_cat = get_df('nlp_data')
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("roberta-base")
