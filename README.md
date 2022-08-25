# IC_Project
This is the repository for the MSc Computing project

## Description of the files
- `trainer.py`: contains our Trainer class
- `loss.py`: contains custom loss functions that we used
- `cleaning.py`: script to clean the Refinitiv raw data
- `data_analysis.py`: contains a class named Preprocessor to process data for the baseline model
- `embedding_viewer`: contains a class named BertEmbeddingView for model interpretation and visualisation
- `data.py`: contains the PyTorch dataset class to feed the inputs
- `model.py`: contains the Model class we used.
- `my_tokenizers`: contains Word-level tokeniser for the LSTM model
- `xgb_train.py`: contains baseline XGBoost model
- `metrics.py`: contains the evluation metrics used in the project
- `FGM.py`: Adverserial attack training techniques
- `utils.py`: contains useful functions for data analsis and manipulation
- `main.py`: the main script that runs the experiment
- `transfer.py`: contains the transfer learning experiment
- `models`: directory contains saved models
- `emb_vis`: directory contains visualisations for model interpretability
- `result`: directory contains results of the test set

## Required packages
`transformers          4.16.2`  
`spacy                 3.2.0`  
`torch                 1.10.0`  
`numpy                 1.19.5`  
`pandas                1.1.5`   
`scikit-learn          0.24.2`  
`tqdm                  4.62.3`  
`xgboost               1.4.2`  
`captum                0.5.0`
`upsetplot             0.6.1`
`beautifulsoup4        4.6.3`
`wordcloud             1.8.2.2`
`seaborn               0.11.2`
`matplotlib            3.2.2`




## How to run the classification
* To replicate the experiment, please first place the data in csv format in the current folder with a column `story` that contains the news stories. Then you can have as many controversial topics as you need, each with an individual column where `1` represent the presence of the controversy.

* Run `cleaning.py` by changing the file name to your placed csv file. The resulted cleaned file is named as `cleaned_2.csv`, of course you can change the name as you wish.

* Please modify `config` varaible in the `main.py` to test different hyperparameters/tricks, or leave it as it is for the best model found in the project.

* To train deep learning model, run `python3 main.py`. To train baseline model, run `python3 xgb_train.py`

## How to run interpretation
* Have your trained model ready, e.g. `models/ProsusAI_finbert_head_3e-06_10_512_False_None_saved_model.pt` in the models folder

* Run `python3 embedding_viewer.py`

## How to run transfer learning
* Have your trained model ready in models folder

* Have target domain data source available, e.g. 'twitter_data.csv'

* Run `python3 transfer.py`
## Important notes
* Please make sure you have all required libraries installed

* `MyBertModel` is the base class the the BERT model implementation. Use `MyBertModel_1` for default classification head and `MyBertModel_2` for the customised classification head.

* The model is saved in `models` directory.

