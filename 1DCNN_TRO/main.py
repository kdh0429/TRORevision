"""
Main experiment
"""
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from configparser import ConfigParser
from datetime import datetime

from model.collision_1d_cnn import Collision1DCNN
from utils.data import Doosan1DCNNDataset
import numpy as np

import wandb

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Friction LSTM')
    parser.add_argument(
        '--globals', type=str, default='./configs/globals.ini', 
        help="Path to the configuration file containing the global variables "
             "e.g. the paths to the data etc. See configs/globals.ini for an "
             "example."
    )
    return parser.parse_args()


def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()

    # Load global variable (e.g. paths)
    config.read(args.globals)

    # Load default model configuration
    default_model_config_filename = config['paths']['model_config_name']
    default_model_config_path = os.path.join(config['paths']['configs_directory'], default_model_config_filename)
    config.read(default_model_config_path)

    config.set('device', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    # config.set('device', 'device', 'cpu')
    return config


def run(config, trainloader, validationloader, testloader, testcollisionloader):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    checkpoint_directory = os.path.join(
        config['paths']['checkpoints_directory'],
        '{}{}/'.format(config['model']['name'], config['model']['config_id']),
        current_time)
    os.makedirs(checkpoint_directory, exist_ok=True)

    collision_net = Collision1DCNN(config, checkpoint_directory, trainloader, validationloader, testloader, testcollisionloader)
    collision_net.to(config['device']['device'])
    collision_net.fit()
    collision_net.test()


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)
    
    if config.getboolean("log", "wandb") is True:
        wandb.init(project="TRO_1dcnn", tensorboard=False)
        wandb_config_dict = dict()
        for section in config.sections():
            for key, value in config[section].items():
                wandb_config_dict[key] = value
        wandb.config.update(wandb_config_dict)

    # Get data path
    data_dir = config.get("paths", "data_directory")

    data_seq_len = config.getint("data", "seqeunce_length")
    data_num_input_feat = config.getint("data", "n_input_feature")
    data_num_output = config.getint("data", "n_output")

    train_data_file_name = config.get("paths", "train_data_file_name")
    train_csv_path = os.path.join(data_dir, train_data_file_name)
    train_data = Doosan1DCNNDataset(train_csv_path,seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output)
    trainloader = DataLoader(
        train_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    validation_data_file_name = config.get("paths", "validation_data_file_name")
    validation_csv_path = os.path.join(data_dir, validation_data_file_name)
    validation_data = Doosan1DCNNDataset(validation_csv_path,seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output)
    validationloader = DataLoader(
        validation_data,
        batch_size=validation_data.__len__(),
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=False)

    test_data_file_name = config.get("paths", "test_data_file_name")
    test_csv_path = os.path.join(data_dir, test_data_file_name)
    test_data = Doosan1DCNNDataset(test_csv_path,seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output)
    testloader = DataLoader(
        test_data,
        batch_size=test_data.__len__(),
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=False)

    test_collision_data_file_name = config.get("paths", "test_collision_data_file_name")
    test_collision_csv_path = os.path.join(data_dir, test_collision_data_file_name)
    test_collision_data = Doosan1DCNNDataset(test_collision_csv_path,seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output)
    test_collision_loader = DataLoader(
        test_collision_data,
        batch_size=test_collision_data.__len__(),
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=False)

    run(config, trainloader, validationloader, testloader,test_collision_loader)