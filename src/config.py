"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed
import time
import json
import logging
import os


def get_config():
    time_prefix = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    prefix = "AVA-cifar10-resnet18"
    config = ed({
        # base setting
        "description": "200 epochs",
        "prefix": prefix,
        "time_prefix":time_prefix,
        "net_work": "resnet18",
        "low_dims": 128,
        "use_MLP": False,

        # save
        "save_checkpoint": True,
        "moxing_save_checkpoint_path": prefix + "/checkpoint" + time_prefix,
        "moxing_summary_path":  prefix + "/summary" + time_prefix,
        "moxing_log_dir": prefix,
        "log_dir": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix,
        "summary_path": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix + "/summary" + time_prefix,
        "save_checkpoint_path": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix + "/checkpoint" + time_prefix,
        "save_checkpoint_epochs": 5,
        "keep_checkpoint_max": 2,
        "moxing_model_save_path": "obs://tuyanlun/model/",
        "load_ckpt_path":"",

        # dataset
        "dataset": "cifar10",
        "moxing_train_data_dir": "cifar-10-batches-bin/train",
        "train_data_dir": "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train",
        "moxing_test_data_dir": "cifar-10-batches-bin/test",
        "test_data_dir": "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/test",

        # optimizer
        "base_lr": 0.03,
        "type": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "loss_scale": 1,
        "sigma":0.1,

        # trainer
        "batch_size": 128,
        "epochs": 200,
        "epoch_stage": [120, 80],
        "lr_schedule": "cosine_lr",
        "lr_mode": "epoch",
        "warmup_epoch": 0,
        "eval_pause":10
    })
    return config

def get_config_linear():
    time_prefix = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    prefix = "AVA-cifar10-linear"
    config = ed({
        # base setting
        "description": "test checkpoint bz1024",
        "prefix": prefix,
        "time_prefix": time_prefix,
        "net_work": "resnet50",
        "low_dims": 128,
        "mid_dims": 2048,

        # save
        "save_checkpoint": True,
        "moxing_save_checkpoint_path": prefix + "/checkpoint" + time_prefix,
        "moxing_summary_path": prefix + "/summary" + time_prefix,
        "moxing_log_dir": prefix,
        "log_dir": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix,
        "summary_path": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix + "/summary" + time_prefix,
        "save_checkpoint_path": "/home/tuyanlun/code/mindspore_r1.0/cifar/" + prefix + "/checkpoint" + time_prefix,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 5,
        "moxing_model_save_path": "obs://tuyanlun/model/",

        # load ckpt if "" then dont load else load with the path
        "load_ckpt_path": "/home/tuyanlun/code/ms_r0.6/project/AVA-cifar10-resnet50/test-resnet50-1000/AVA-994_0.9191_391.ckpt",
        "load_ckpt_path_moxing":"obs://tuyanlun/model/checkpoint-20200807-161317",
        "load_ckpt_filename":"AVA-994_0.9191_391.ckpt",
        # dataset
        "num_classes":10,
        "dataset": "cifar10",
        "moxing_train_data_dir": "cifar-10-batches-bin/train",
        "train_data_dir": "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train",
        "moxing_test_data_dir": "cifar-10-batches-bin/test",
        "test_data_dir": "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/test",

        # optimizer
        "base_lr": 0.01,
        "type": "Adam",
        "beta1": 0.5,
        "beta2": 0.999,
        "weight_decay": 0,
        "loss_scale": 1,

        # trainer
        "batch_size": 128,
        "epochs": 50,
        "epoch_stage": [30, 20],
        "lr_schedule": "cosine_lr",
        "lr_mode": "epoch"
    })
    return config

def save_config(paths, dict):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        file = open(path, "w")
        json.dump(dict, file, indent=4)
        file.close()


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w+")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
