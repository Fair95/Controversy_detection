#!/vol/bitbucket/ql4618/myvenv/bin/python

import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from xgboost import XGBClassifier
from utils import save, load

from data_analysis import Preprocessor
from metrics import MyMetric

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

if __name__=='__main__':

    from utils import get_df
    from cleaning import run_cleaning

    news_data = 'second_test.csv'
    cleaned_data = 'cleaned_2.csv'
    run_cleaning(raw_data_path=news_data,
                 max_len=5000, min_len=200,
                 save_path=cleaned_data)
    
    df_train, df_val, df_test = get_df(cleaned_data)

    pre = Preprocessor()
    df_train_val = pd.concat([df_train, df_val])
    X_train, y_train = pre.get_tfidf_vectors(df=df_train_val, load_path='new_spacy_preprocessed', train=True)
    # X_val, y_val = pre.get_tfidf_vectors(df=df_val, train=False)
    X_test, y_test = pre.get_tfidf_vectors(df=df_test, load_path='new_spacy_preprocessed', train=False)


    xgb_params = {'estimator__eta': [0.001, 0.01, 0.05, 0.1, 0.2],
                      'estimator__max_depth': [3, 6, 9, 12, 15], 
                      'estimator__subsample': [0.8], 
                      'estimator__colsample_bytree': [0.5, 0.75],
                      'estimator__n_estimators': [100, 500, 1000],
                      "estimator__gamma":[0, 0.5, 1],
                      #'min_child_weight' : 1.5,
                      # 'scale_pos_weight': imbalance_weight,
                      'estimator__objective': ['binary:logistic'], 
                      # 'estimator__eval_metric': ['logloss'],
                      'estimator__seed': [23],
                      'estimator__lambda': [1.5],
                      'estimator__alpha': [0.6]
                    }
    xgb_classifier = OneVsRestClassifier(XGBClassifier(**xgb_params))

    model_name = 'xgb_model.pkl'
    try:
        print('[Load pretrained model...]')
        clf = load(model_name)
    except FileNotFoundError:
        print('[No pretrained model found...]')
        print('[Start fitting new model]...')
        clf = HalvingGridSearchCV(xgb_classifier, param_grid=xgb_params, factor=3,
            resource='n_samples',
            scoring='f1_micro', cv=5, n_jobs=-1, verbose=3)
        clf.fit(X_train, y_train)
        save(clf, model_name)
    print(['Evaluating...'])
    y_pred = clf.predict(X_test)
    metric = MyMetric()
    metric.update(y_pred, y_test)
    print(metric)
