import os
import argparse
import random
import numpy as np
import time
import logging

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# from mindspore.nn import SGD
from src.optimizer import SGD_ as SGD
import mindspore.dataset.engine as de
from mindspore.train.callback import SummaryCollector

from src.config import get_config, save_config, get_logger
from src.datasets import get_train_dataset, get_test_dataset, get_train_test_dataset
from src.cifar_resnet import resnet18, resnet50, resnet101
from src.network_define import WithLossCell, TrainOneStepCell
from src.callbacks import LossCallBack, ModelCheckpoint_
from src.loss import LossNet
from src.lr_schedule import step_cosine_lr,cosine_lr
from src.knn_eval import KnnEval, FeatureCollectCell

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA training")
parser.add_argument("--use_moxing", type=bool, default=False, help="whether use moxing for huawei cloud.")
parser.add_argument("--data_url", type=str, default='', help="huawei cloud ModelArts need it.")
parser.add_argument("--train_url", type=str, default='', help="huawei cloud ModelArts need it.")
parser.add_argument("--src_url", type=str, default='obs://tuyanlun/data/cifar10', help="huawei cloud ModelArts need it.")
parser.add_argument("--dst_url", type=str, default='/cache/data', help="huawei cloud ModelArts need it.")
parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
parser.add_argument("--do_train", type=bool, default=True, help="Do train or not, default is true.")
parser.add_argument("--do_eval", type=bool, default=False, help="Do eval or not, default is false.")
parser.add_argument("--pre_trained", type=str, default="", help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=5, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
#parser.add_argument("--mindspore_version", type=float, default=0.6, help="Mindspore version default 0.6.")
args_opt = parser.parse_args()



if __name__ == '__main__':
    config = get_config()
    temp_path = ''
    if args_opt.use_moxing:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        print("get path mapping with huawei cloud...")
        import moxing as mox

        temp_path = args_opt.dst_url
    else:
        print("do not use moxing for huawei cloud...")
        device_id = args_opt.device_id
        device_num = args_opt.device_num

    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)
    
    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=False, parameter_broadcast=True)
        init()
        temp_path = os.path.join(temp_path, str(device_id))
        print("temp path with multi-device:{}".format(temp_path))

    if args_opt.use_moxing:
        mox.file.shift('os', 'mox')
        print("data url:{}".format(args_opt.data_url))
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=temp_path)

        checkpoint_dir = os.path.join(temp_path, config.moxing_save_checkpoint_path)
        summary_dir = os.path.join(temp_path, config.moxing_summary_path)
        train_data_dir = os.path.join(temp_path, config.moxing_train_data_dir)
        test_data_dir = os.path.join(temp_path, config.moxing_test_data_dir)
        log_dir = os.path.join(temp_path, config.moxing_log_dir)
    else:
        checkpoint_dir = os.path.join(temp_path, config.save_checkpoint_path)
        summary_dir = os.path.join(temp_path, config.summary_path)
        train_data_dir = os.path.join(temp_path, config.train_data_dir)
        test_data_dir = os.path.join(temp_path, config.test_data_dir)
        log_dir = os.path.join(temp_path, config.log_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = get_logger(os.path.join(log_dir, 'log' + config.time_prefix + '.log'))

    print("start create dataset...")

    train_dataset = get_train_dataset(train_data_dir=train_data_dir, batchsize=config.batch_size,
                                      epoch=1, device_id=device_id, device_num=device_num)
    train_dataset.__loop_size__ = 1

    # eval_dataset contains train dataset and test dataset, which is used for knn eval
    eval_dataset = get_train_test_dataset(train_data_dir=train_data_dir, test_data_dir=test_data_dir,
                                          batchsize=100, epoch=1)

    train_dataset_batch_num = int(train_dataset.get_dataset_size())
    eval_dataset_batch_num = int(eval_dataset.get_dataset_size())

    print("the chosen network is {}".format(config.net_work))
    logger.info("the chosen network is {}".format(config.net_work))

    if config.net_work == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif config.net_work == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif config.net_work == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    else:
        raise ("net work config error!!!")

    # logger.info(resnet)
    # eval_network = FeatureCollectCell(resnet)

    loss = LossNet(temp=config.sigma)

    net_with_loss = WithLossCell(resnet, loss)

    # eval_network = FeatureCollectCell(net_with_loss)
    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode,
            warmup_epoch=config.warmup_epoch
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode,
            warmup_epoch=config.warmup_epoch
        ), mstype.float32)

    opt = SGD(params=net_with_loss.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if device_num > 1:
        net = TrainOneStepCell(net_with_loss, opt, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt)

    eval_network = FeatureCollectCell(resnet)

    loss_cb = LossCallBack(data_size=train_dataset_batch_num)
    # summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)
    # cb = [loss_cb, summary_collector]
    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint_(prefix='AVA',
                                      directory=checkpoint_dir,
                                      config=ckptconfig)
        cb += [ckpoint_cb]

    # model = Model(net,metrics={'knn_acc':KnnEval(batch_size=config.batch_size,device_num=1)},eval_network=eval_network)
    model = Model(net, metrics={'knn_acc': KnnEval(batch_size=config.batch_size, device_num=1)},
                  eval_network=eval_network)

    # model.init(train_dataset, eval_dataset)
    # model =Model(net)

    logger.info("save configs...")
    print("save configs...")
    # save current config
    config_name = 'config.json'
    save_config([os.path.join(checkpoint_dir, config_name)], config)

    logger.info("training begins...")
    print("training begins...")

    logger.info("model description:{}".format(config.description))
    print("model description:{}".format(config.description))
    best_acc = 0
    for epoch_idx in range(1, config.epochs + 1):
        model.train(1, train_dataset, callbacks=cb, dataset_sink_mode=True)
        eval_start = time.time()
        if epoch_idx % config.eval_pause == 0 or epoch_idx == 1 or epoch_idx >= config.epochs - 10:
            output = model.eval(eval_dataset)
            knn_acc = float(output["knn_acc"])
        else:
            knn_acc = 0
        ckpoint_cb.set_epoch(epoch_idx)
        ckpoint_cb.set_force_to_save(False)
        if knn_acc > best_acc:
            best_acc = knn_acc
            ckpoint_cb.set_force_to_save(True)
            ckpoint_cb.set_acc(best_acc)

        eval_cost = time.time() - eval_start
        time_cost = loss_cb.get_per_step_time()
        loss = loss_cb.get_loss()
        print("the {} epoch's resnet result: "
              " training loss {}, knn_acc {}, "
              "training per step cost {:.2f} s, eval cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, loss, knn_acc, time_cost, eval_cost, time_cost * train_dataset_batch_num + eval_cost))
        logger.info("the {} epoch's resnet result: "
                    " training loss {}, knn_acc {}, "
                    "training per step cost {:.2f} s, eval cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, loss, knn_acc, time_cost, eval_cost, time_cost * train_dataset_batch_num + eval_cost))

    if args_opt.use_moxing:
        print("download file to obs...")
        mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
                               dst_url=os.path.join(config.moxing_model_save_path, config.prefix))

