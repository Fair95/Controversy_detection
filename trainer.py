import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from loss import FocalLoss
import pandas as pd
import os
from FGM import FGM
from metrics import MyMetric
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class Trainer(object):
    """
    Our trainer of BERT model.
    """
    def __init__(self, model, config=None, train_loader=None, val_loader=None):
        if config is None:
            print('!!! You are initialising trainer with no training config !!!\n \
                   !!! Expect Error if you try to train model !!!')
            self.model = model
            self.metric = MyMetric(threshold=0.5)
        else:
            self.config = config
            self.epochs = self.config['epochs']
            self.gas = self.config['gradient_accumulate_steps']
            self.lr = self.config['lr']
            self.use_layerwise_learning_rate = self.config['use_layerwise_learning_rate']

            self.model = model.to(device)
            if self.use_layerwise_learning_rate:
                self.optimizer = self.get_opt_with_layerwise_learning_rate()
            else:
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr)
            self.loss_fn = nn.BCEWithLogitsLoss()
            # self.loss_fn = FocalLoss(reduction="mean")
            if train_loader is not None:
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=0, # Default value
                    num_training_steps=self.epochs * (len(train_loader)//self.gas) # Note the traing steps also adjust based on gas
                )
                self.train_loader = train_loader 
            if val_loader is not None:
                self.val_loader = val_loader 
            self.metric = MyMetric(threshold=0.5)

            self.fgm = FGM(self.model)

            if 'lstm' in self.model.name:
                self.save_path ='models/{}_{}_{}_{}_{}_{}_{}saved_model.pt'.format(
                            self.model.name.replace('/','_'), str(self.config['lr']), str(self.config['input_max_length']), 
                            str(self.config['use_attention']), str(self.config['bidirectional']),
                            str(self.config['dropout_rate']), str(self.config['num_layers']))
            else:
                self.save_path ='models/{}_{}_{}_{}_{}_{}_saved_model.pt'.format(
                            self.model.name.replace('/','_'), str(self.config['lr']), str(self.config['epochs']), 
                            str(self.config['input_max_length']), str(self.config['use_layerwise_learning_rate']),
                            str(self.config['resample_scale']))


    def get_opt_with_layerwise_learning_rate(self):
        no_decay = ['bias', 'LayerNorm']
        group1=['layer.0.','layer.1.','layer.2.','layer.3.']
        group2=['layer.4.','layer.5.','layer.6.','layer.7.']
        group3=['layer.8.','layer.9.','layer.10.','layer.11.']
        group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
        optimizer_parameters = [
          {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
          {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': self.lr/2.6},
          {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': self.lr},
          {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': self.lr*2.6},
          {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
          {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': self.lr/2.6},
          {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': self.lr},
          {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': self.lr*2.6},
        ]
        return torch.optim.AdamW(params=optimizer_parameters, lr=self.lr)

    def from_checkpoint(self, model_path='models/saved_model.pt'):
        """
        Function to load trained model
        """
        if os.path.exists(model_path):
            print('-'*60)
            print('Loading pretrained model:{}...'.format(model_path))
            print('-'*60)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print('-'*60)
            print('Success!')
            print('-'*60)
        else:
            print('-'*60)
            print('Provided path {} not found!'.format(model_path))
            print('-'*60)
    
    def save_checkpoint(self, save_path='models/saved_model.pt'):
        torch.save(self.model.state_dict(), save_path)
        print('-'*60)
        print('Model saved as path {}!'.format(save_path))
        print('-'*60)

    def run_one_epoch(self, loader, logging_freq, eval=False ):
        """
        Fuction to train for one epoch
        loader: dataloader
        logging_freq: how many steps do we log the running stat
        eval: whether running train or evaluation
        """
        # Moving average statistics
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        batch_loss = []
        batch_accuracy = []

        all_y_true = []
        all_y_pred = []

        # all_hard_examples = pd.DataFrame()
        
        data_iter = tqdm(enumerate(loader), total=len(loader), bar_format="{bar}{l_bar}{r_bar}")
        for step, batch in data_iter:
            batch = [x.to(device) for x in batch]
            y_true = batch[-1]
            y_pred = self.model(batch[:-1])

            # Training, actively update parameters
            if not eval:
                self.optimizer.zero_grad()
                # Normalise the loss if accumulation applied (decouple from learning rate)
                loss = self.loss_fn(y_pred, y_true)/self.gas
                loss.backward()

                # self.fgm.attack() 
                # y_pred_adv = self.model(batch[:-1])
                # loss_adv = self.loss_fn(y_pred_adv, y_true)/self.gas
                # loss_adv.backward()
                # self.fgm.restore()
                # Accumulate gradient to effectively increase batch size
                if step % self.gas == 0:
                    self.optimizer.step()
                    self.scheduler.step()


                # Since F1 only calculated per whole Epoch
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

                self.metric.update(y_pred, y_true)
                loss = loss.cpu().item()
                
                batch_loss.append(loss)
                # batch_accuracy.append(accuracy)

            # Evaluation, no gradient
            else:
                loss = self.loss_fn(y_pred, y_true)

                # Since F1 only calculated per whole Epoch
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
                
                self.metric.update(y_pred, y_true)

                loss = loss.cpu().item()

                batch_loss.append(loss)
                # batch_accuracy.append(accuracy)

                # hard_examples = self.hard_sample_mining(y_pred, y_true, batch[0])
                # hard_examples = pd.DataFrame.from_dict(hard_examples)
                # all_hard_examples = all_hard_examples.append(hard_examples, ignore_index=True)
                

            # Logging statistics
            mode = "Train" if not eval else "Eval"
            # Moving average of batches
            if (step+1) % logging_freq == 0: # Use 1-based index for logging
                data_iter.set_description("[Batch {} Running Stat] Mode: {} | Step: {} | Loss: {} | Metric: {}".format(
                    logging_freq, mode, step + 1,

                    np.mean(batch_loss[(step + 1 - logging_freq) : (step+1)]), 
                    self.metric.cal_running_statistics(peroid=logging_freq)
                    # np.mean(batch_accuracy[(step + 1 - logging_freq) : (step+1)])
                )
            )
            # Whole average of batch in one epoch
            if (step) == len(loader)-1:
                data_iter.set_description("[Per Batch Stat] Mode: {} | End of Epoch | Loss: {} | Metric: {}".format(
                    mode, 
                    np.mean(batch_loss),
                    # np.mean(batch_accuracy),
                    self.metric.cal_running_statistics()
                )
            )
        epoch_loss = np.mean(batch_loss)
        epoch_accuracy = self.metric.cal_running_statistics()
        epoch_f1 = self.metric.cal_running_statistics(method='micro_f1')

        # if eval:
        #     all_hard_examples.to_csv('all_hard_examples.csv')
        return epoch_loss, epoch_accuracy, epoch_f1
    
    def train(self, logging_freq=10, val_freq=20):
        """
        Function to train the model
        logging_freq: how many steps do we log the running stat
        val_freq: frequency to do evaluation
        """
        # Moving average statistics
        train_loss = 0.0
        train_accuracy = 0.0
        train_f1 = 0.0
        val_loss = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0

        best_f1 = 0.0

        for i in range(self.epochs):
            self.model.train()
            self.metric.reset()
            print('-' * 30 + 'Train for Epoch {}'.format(i+1) + '-'*30 )
            epoch_loss, epoch_accuracy, epoch_f1 = self.run_one_epoch(
                self.train_loader, 
                logging_freq=logging_freq, 
                eval=False
            )
            
            train_loss += epoch_loss
            train_accuracy += epoch_accuracy
            train_f1 += epoch_f1

            print("[Epoch Running Stat] Mode: Train | Epoch: {} | Loss: {} | Metric: {} | F1: {}".format(
                  i + 1, train_loss / (i+1), train_accuracy / (i+1), epoch_f1
                )
            )

            # Do evaluation 
            if i % val_freq == 0:
                self.model.eval()
                self.metric.reset()
                with torch.no_grad():
                    print('-' * 30 + 'Val at Epoch{}'.format(i+1) + '-'*30 )
                    epoch_loss, epoch_accuracy, epoch_f1 = self.run_one_epoch(
                        self.val_loader, 
                        logging_freq=logging_freq, 
                        eval=True
                    )

                    val_loss += epoch_loss
                    val_accuracy += epoch_accuracy
                    val_f1 += epoch_f1
                    print("[Epoch Running Stat] Mode: Eval | Epoch: {} | Loss: {} | Metric: {} | F1: {}".format(
                        i//val_freq + 1, val_loss / (i//val_freq+1), val_accuracy / (i//val_freq+1), epoch_f1
                        )
                    )

                    # Save best model
                    if epoch_f1 > best_f1 :
                        self.save_checkpoint(self.save_path)
                        best_f1 = epoch_f1
                print(self.metric)

    def inference(self, data):
        with torch.no_grad():
            data = [x.to(device) for x in data]
            y_pred = self.model(data)
            sigmoid = torch.nn.Sigmoid()
            y_pred = sigmoid(y_pred)
            y_pred = 1 * (y_pred > self.metric.threshold)
            return y_pred.cpu().numpy()[0].tolist()

