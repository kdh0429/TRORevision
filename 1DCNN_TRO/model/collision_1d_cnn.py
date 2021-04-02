#!/usr/bin/python3
"""
1D CNN for Collision Detection
"""
from itertools import chain
import time
import json
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
import wandb

class Collision1DCNN(nn.Module):
    def __init__(self, config, checkpoint_directory, trainloader, validationloader, testloader, testcollisionloader):
        super(Collision1DCNN, self).__init__()
        self.config = config
        
        self._device = config['device']['device']
        self.num_epochs = config.getint('training', 'n_epochs')
        self.cur_epoch = 0
        self.checkpoint_directory = checkpoint_directory
        self._save_every = config.getint('model', 'save_every')
        self.model_name = '{}{}'.format(config['model']['name'], config['model']['config_id'])

        if self.config.getboolean("log", "wandb") is True:
            self.wandb_dict = dict()

        self.trainLoader = trainloader
        self.validationLoader = validationloader
        self.testLoader = testloader
        self.testCollisionLoader = testcollisionloader

        config_collision_net = json.loads(config.get("model", "structure"))

        collision_network = []
        for layer in config_collision_net:
            if layer['type'] == 'linear':
                collision_network.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'conv1d':
                collision_network.append(nn.Conv1d(layer['in_channels'], layer['out_channels'], layer['kernel_size'], padding=layer['padding']))
            elif layer['type'] == 'relu':
                collision_network.append(nn.ReLU())
            elif layer['type'] == 'flatten':
                collision_network.append(nn.Flatten())

        self.collision_network = nn.Sequential(*collision_network)

        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

    def forward(self, X):
        return self.collision_network(X)

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1

            if self.config.getboolean("log", "wandb") is True:
                self.wandb_dict.clear()

            # temporary storage
            train_losses = []
            train_accuracy = []
            batch = 0
            for inputs, outputs in self.trainLoader:
                self.train()
                self._optim.zero_grad()
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                predictions = self.forward(inputs)
                train_loss = nn.BCEWithLogitsLoss(reduction='mean')(predictions, outputs)
                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
                train_accuracy.append(accuracy_score(np.argmax(self._to_numpy(outputs),axis=1), np.argmax(self._to_numpy(predictions),axis=1)))
                batch += 1
            
            if self.config.getboolean("log", "wandb") is True:
                self.wandb_dict['Training Accuracy'] = np.mean(train_accuracy)
                self.wandb_dict['Training Loss'] = np.mean(train_losses)

            print('Training Accuracy: ', np.mean(train_accuracy))
            print('Training Loss: ', np.mean(train_losses))

            self.evaluate(self.validationLoader)

            if epoch % 100 == 0:
                self.test()

            if epoch % self._save_every == 0:
                self.save_checkpoint()

            if self.config.getboolean("log", "wandb") is True:
                wandb.log(self.wandb_dict)



        self.save_checkpoint()

    def test(self):
        print("--------------------------- TEST ---------------------------")
        self.eval()

        # Free motion Evaluation
        for inputs, outputs in self.testLoader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            test_loss = nn.BCEWithLogitsLoss(reduction='mean')(preds, outputs)

            false_positive_local_arr = np.zeros((len(outputs),1))
            for j in range(len(false_positive_local_arr)):
                false_positive_local_arr[j] = self._to_numpy(preds[j,0]) > 0.8 and np.equal(np.argmax(self._to_numpy(outputs[j,:])), 1)
            print('False Positive: ', sum(false_positive_local_arr))
            print('Total Num: ', len(outputs))

        # Collision
        for inputs, outputs in self.testCollisionLoader:
            inputs = inputs.to(self._device)
            preds = self.forward(inputs)
            preds = self._to_numpy(preds)

            prediction = np.argmax(preds, 1)
            t = np.arange(0,0.001*len(prediction),0.001)

            collision_pre = 0
            collision_cnt = 0
            collision_time = 0
            detection_time_NN = []
            collision_status = False
            NN_detection = False
            collision_fail_cnt_NN = 0

            for i in range(len(prediction)):
                if (outputs[i,0] == 1 and collision_pre == 0):
                    collision_cnt = collision_cnt +1
                    collision_time = t[i]
                    collision_status = True
                    NN_detection = False
                
                if (collision_status == True and NN_detection == False):
                    if(preds[i,0] > 0.8):
                        NN_detection = True
                        detection_time_NN.append(t[i] - collision_time)

                if (outputs[i,0] == 0 and collision_pre == 1):
                    collision_status = False
                    if(NN_detection == False):
                        detection_time_NN.append(0.0)
                        collision_fail_cnt_NN = collision_fail_cnt_NN+1
                collision_pre = outputs[i,0]

            if self.config.getboolean("log", "wandb") is True:
                self.wandb_dict['False Positive'] = sum(false_positive_local_arr)
                self.wandb_dict['NN Failure'] = collision_fail_cnt_NN
                self.wandb_dict['NN Detection Time'] = sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN)

            print('Total collision: ', collision_cnt)
            print('NN Failure: ', collision_fail_cnt_NN)
            print('NN Detection Time: ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))

    def evaluate(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        validation_losses = []
        validation_accuracy = []
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            validation_loss = nn.BCEWithLogitsLoss(reduction='mean')(preds, outputs)
            validation_losses.append(self._to_numpy(validation_loss))
            validation_accuracy.append(accuracy_score(np.argmax(self._to_numpy(outputs),axis=1), np.argmax(self._to_numpy(preds),axis=1)))

        if self.config.getboolean("log", "wandb") is True:
            self.wandb_dict['Validation Accuracy'] = np.mean(validation_accuracy)
            self.wandb_dict['Validation Loss'] = np.mean(validation_losses)

        print("Validation Accuracy: ", np.mean(validation_accuracy))
        print("Validation Loss: ", np.mean(validation_losses))




    def save_checkpoint(self):
        """Save model paramers under config['model_path']"""
        model_path = '{}/epoch_{}.pt'.format(
            self.checkpoint_directory,
            self.cur_epoch)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, epoch):
        """
        Retore the model parameters
        """
        model_path = '{}{}_{}.pt'.format(
            self.config['paths']['checkpoints_directory'],
            self.model_name,
            epoch)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_epoch = epoch
