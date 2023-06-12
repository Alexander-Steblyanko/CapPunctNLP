import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import transformers
import os
import sentencepiece
from numpy import inf
from tqdm import tqdm
import torch_optimizer

from pdata import *
from model import *
from settings import *


class TrainEnv():
    def __init__(self):
        # Technical
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Model
        self.model = PunctuationModel(bert_model_name, freeze_bert=freeze)
        self.model.to(device)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        # self.optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=learning_rate, weight_decay=decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=decay)
        self.optimizer = torch_optimizer.Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True,
                                                   warmup_init=True, lr=None)
        # self.scheduler = transformers.AdafactorSchedule(self.optimizer)
        #self.scheduler = torch.optim.lr_scheduler.CyclicLR \
        #    (self.optimizer, base_lr=learning_rate, max_lr=max_rate, step_size_up=100, mode="triangular")

        # Datasets
        data_set = Dataset(dataset_path)
        print("Data set ready!")
        train_set, val_set, test_set = torch.utils.data.random_split(data_set, [0.90, 0.08, 0.02])
        print("Subsets ready!")

        # Data Loaders
        data_loader_params = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers
        }
        self.train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
        self.val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
        self.test_loader = torch.utils.data.DataLoader(test_set, **data_loader_params)
        print("Loaders ready!")

    def validate(self):
        """
         :return: validation accuracy, validation loss
        """
        num_iteration = 0
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for x, y, att, y_mask in self.val_loader:
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)

                y_predict = self.model(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = self.criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                val_loss += loss.item()
                num_iteration += 1
                y_mask = y_mask.view(-1)
                correct += torch.sum(y_mask * (y_predict == y)).item()
                total += torch.sum(y_mask).item()
        return correct / total, val_loss / num_iteration

    def test(self):
        """
        :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
        """
        num_iteration = 0
        self.model.eval()
        # +1 for overall result
        true_positive = np.zeros(1 + len(punc_dict), dtype=int)
        false_positive = np.zeros(1 + len(punc_dict), dtype=int)
        false_negative = np.zeros(1 + len(punc_dict), dtype=int)
        conf_matrix = np.zeros((len(punc_dict), len(punc_dict)), dtype=int)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, att, y_mask in tqdm(self.test_loader, desc='test'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)

                y_predict = self.model(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                num_iteration += 1
                y_mask = y_mask.view(-1)
                correct += torch.sum(y_mask * (y_predict == y)).item()
                total += torch.sum(y_mask).item()

                for i in range(y.shape[0]):
                    if y_mask[i] == 0:
                        # we can ignore this because we know there won't be any punctuation in this position
                        # since we created this position due to padding or sub-word tokenization
                        continue
                    cor = y[i]
                    prd = y_predict[i]
                    if cor == prd:
                        true_positive[cor] += 1
                    else:
                        false_negative[cor] += 1
                        false_positive[prd] += 1
                    conf_matrix[cor][prd] += 1

        # ignore first index which is for no punctuation
        true_positive[-1] = np.sum(true_positive[1:])
        false_positive[-1] = np.sum(false_positive[1:])
        false_negative[-1] = np.sum(false_negative[1:])
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)

        #print(precision, recall, f1, correct / total, conf_matrix)
        return precision, recall, f1, correct / total, conf_matrix

    def train(self):
        best_val_loss = inf

        self.model.load_state_dict(torch.load(model_load_path))

        # Accounts for autocast usage
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch} begins!")
            train_loss = 0
            train_iteration = 0
            correct = 0
            total = 0
            self.model.train()

            if epoch == 1:
                for p in self.model.bert_layer.parameters():
                    p.requires_grad = False

            for x, y, x_mask, y_mask in tqdm(self.train_loader, desc='train'):
                # ~6 secs per batch if no freeze, ~2 secs if freeze
                # print(train_iteration*batch_size)
                self.optimizer.zero_grad(set_to_none=True)
                # Send data to GPU
                x, y, x_mask, y_mask = x.to(device), y.to(device), x_mask.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)

                with torch.cuda.amp.autocast():
                    y_predict = self.model(x, x_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y = y.view(-1)
                    loss = self.criterion(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)

                correct += torch.sum(y_mask * (y_predict == y)).item()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()
                train_iteration += 1
                total += torch.sum(y_mask).item()

                #self.scheduler.step()

                # del x, y, x_mask, y_mask, y_predict, loss
                torch.cuda.empty_cache()

            train_loss /= train_iteration
            log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
            print(log)

            val_acc, val_loss = self.validate()
            log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
            print(log)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                print("Model saved!")

        print('Best validation Loss:', best_val_loss)
        self.model.load_state_dict(torch.load(model_save_path))
        precision, recall, f1, accuracy, conf_matrix = self.test()
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
              '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix:\n' + str(conf_matrix) + '\n'
        print(log)


if __name__ == '__main__':
    te = TrainEnv()
    te.train()
