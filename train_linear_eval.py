import os
import argparse
import random
import numpy as np
import time
import logging
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.dataset.engine as de
from mindspore.train.callback import SummaryCollector
from mindspore.nn import Adam, SGD

from callbacks import LossCallBack, ModelCheckpoint_0_5, ModelCheckpoint_0_6
from lr_schedule import step_cosine_lr, cosine_lr, constant_lr
from mindspore.ops import operations as P
from mindspore.communication.management import GlobalComm

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA training")
parser.add_argument("--use_moxing", type=bool, default=False, help="whether use moxing for huawei cloud.")
parser.add_argument("--data_url", type=str, default='', help="huawei cloud ModelArts need it.")
parser.add_argument("--train_url", type=str, default='', help="huawei cloud ModelArts need it.")
parser.add_argument("--src_url", type=str, default='s3://tuyanlun/data/', help="huawei cloud ModelArts need it.")
parser.add_argument("--dst_url", type=str, default='/cache/data', help="huawei cloud ModelArts need it.")
parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
parser.add_argument("--do_train", type=bool, default=True, help="Do train or not, default is true.")
parser.add_argument("--do_eval", type=bool, default=False, help="Do eval or not, default is false.")
parser.add_argument("--pre_trained", type=str, default="", help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
parser.add_argument("--mindspore_version", type=float, default=0.6, help="Mindspore version default 0.6.")
parser.add_argument("--imagenet_eval", type=bool, default=False, help="whether eval model on imagenet")
args_opt = parser.parse_args()

if args_opt.imagenet_eval:
    print("linear training on imagenet!")
    from config_imagenet import get_config, save_config, get_logger, get_config_linear
    from imagenet_dataset import get_train_dataset, get_test_dataset
    from imagenet_resnet import resnet18_linear, resnet50_linear, resnet101_linear
else:
    print("linear training on cifar10!")
    from config import get_config, save_config, get_logger, get_config_linear
    from datasets import get_train_dataset, get_test_dataset
    from cifar_resnet import resnet18_linear, resnet50_linear, resnet101_linear

# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class linear_layer_cell(nn.Cell):
    def __init__(self, mid_dims=512, num_classes=10):
        super(linear_layer_cell, self).__init__()
        self.end_point_1 = _fc(mid_dims, num_classes)

    def construct(self, x):
        x = self.end_point_1(x)
        return x


class linear_layer_resnet_cell(nn.Cell):
    def __init__(self, resnet, linear_layer):
        super(linear_layer_resnet_cell, self).__init__()
        self.resnet = resnet
        self.linear_layer = linear_layer

    def construct(self, x):
        # x = self.resnet(x, x, x)
        x = self.resnet(x)
        x = self.linear_layer(x)
        return x


# class linear_layer_resnet_eval_cell(nn.Cell):
#     def __init__(self, linear_layer_resnet):
#         super(linear_layer_resnet_eval_cell, self).__init__()
#         self.linear_layer_resnet = linear_layer_resnet
#
#     def construct(self, x, label):
#         x = self.linear_layer_resnet(x)
#         return x

class ClassifyCorrectCell(nn.Cell):
    r"""
    Cell that returns correct count of the prediction in classification network.
    This Cell accepts a network as arguments.
    It returns orrect count of the prediction to calculate the metrics.
    Args:
        network (Cell): The network Cell.
    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
    Outputs:
        Tuple, containing a scalar correct count of the prediction
    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> eval_net = nn.ClassifyCorrectCell(net)
    """

    def __init__(self, network):
        super(ClassifyCorrectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        return (y_correct,)


class DistAccuracy(nn.Metric):
    r"""
    Calculates the accuracy for classification data in distributed mode.
    The accuracy class creates two local variables, correct number and total number that are used to compute the
    frequency with which predictions matches labels. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides correct number by total number.
    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}
    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for classification (single-label).
    Examples:
        >>> y_correct = Tensor(np.array([20]))
        >>> metric = nn.DistAccuracy(batch_size=3, device_num=8)
        >>> metric.clear()
        >>> metric.update(y_correct)
        >>> accuracy = metric.eval()
    """

    def __init__(self, batch_size, device_num):
        super(DistAccuracy, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.
        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type
        Raises:
            ValueError: If the number of the input is not 1.
        """

        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        # self._total_num += self.batch_size * self.device_num
        self._total_num += self.batch_size

    def eval(self):
        """
        Computes the accuracy.
        Returns:
            Float, the computed result.
        Raises:
            RuntimeError: If the sample size is 0.
        """

        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num


if __name__ == '__main__':
    config = get_config_linear()
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

    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True, parameter_broadcast=True)
        init()
        temp_path = os.path.join(temp_path, str(device_id))
        print("temp path with multi-device:{}".format(temp_path))

    if args_opt.use_moxing:
        mox.file.shift('os', 'mox')
        #mox.file.copy_parallel(src_url=os.path.join(args_opt.data_url, 'val'), dst_url=os.path.join(temp_path, 'val'))
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
    # if not os.path.exists(summary_dir):
    #     os.makedirs(summary_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = get_logger(os.path.join(log_dir, 'log' + config.time_prefix + '.log'))

    print("start create dataset...")

    epoch_for_dataset = config.epochs if args_opt.mindspore_version == 0.5 else 1

    train_dataset = get_train_dataset(train_data_dir=train_data_dir, batchsize=config.batch_size,
                                      epoch=epoch_for_dataset,
                                      mode="linear_eval", device_id=device_id, device_num=device_num)
    # train_dataset.__loop_size__ = 1

    eval_dataset = get_test_dataset(test_data_dir=test_data_dir, batchsize=50, epoch=epoch_for_dataset,
                                    device_id=device_id, device_num=device_num)

    train_dataset_batch_num = train_dataset.get_dataset_size()
    eval_dataset_batch_num = eval_dataset.get_dataset_size()

    print("the chosen network is {}".format(config.net_work))
    logger.info("the chosen network is {}".format(config.net_work))

    if config.net_work == 'resnet18':
        resnet = resnet18_linear(low_dims=config.low_dims, training_mode=False)
    elif config.net_work == 'resnet50':
        resnet = resnet50_linear(low_dims=config.low_dims, training_mode=False)
    elif config.net_work == 'resnet101':
        resnet = resnet101_linear(low_dims=config.low_dims, training_mode=False)
    else:
        raise ("net work config error!!!")

    if config.load_ckpt_path == '':
        raise ("error ckpt path is empty!!!")

    if args_opt.use_moxing:
        mox.file.copy_parallel(src_url=config.load_ckpt_path_moxing, dst_url='/cache/model')
        load_checkpoint(os.path.join('/cache/model', config.load_ckpt_filename), net=resnet)
        print("load ckpt from {}".format(config.load_ckpt_path_moxing))
        logger.info("load ckpt from {}".format(config.load_ckpt_path_moxing))
    else:
        load_checkpoint(config.load_ckpt_path, net=resnet)
        print("load ckpt from {}".format(config.load_ckpt_path))
        logger.info("load ckpt from {}".format(config.load_ckpt_path))

    loss = SoftmaxCrossEntropyWithLogits(reduction="mean", sparse=True, num_classes=config.num_classes)

    # resnet.set_train(mode=False)

    linear_layer = linear_layer_cell(mid_dims=config.mid_dims, num_classes=config.num_classes)
    linear_layer_resnet = linear_layer_resnet_cell(resnet=resnet, linear_layer=linear_layer)
    # linear_layer_resnet_eval = linear_layer_resnet_eval_cell(linear_layer_resnet)
    print(linear_layer_resnet)

    # input=Tensor(np.random.rand(128,3,32,32),mstype.float32)
    # input1 = Tensor(np.random.rand(128, 3, 32, 32),mstype.float32)
    # input2 = Tensor(np.random.rand(128, 3, 32, 32),mstype.float32)
    # print(P.Shape()(linear_layer_resnet(input)))
    eval_network = ClassifyCorrectCell(linear_layer_resnet)

    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    group_params = [{'params': linear_layer.trainable_params()},
                    {'params': resnet.trainable_params(),
                     'lr': Tensor(constant_lr(0, config.epochs, train_dataset_batch_num), mstype.float32)}]

    opt = Adam(params=group_params, learning_rate=lr, loss_scale=config.loss_scale, beta1=config.beta1,
               beta2=config.beta2)
    # opt = SGD(params=group_params, learning_rate=lr,momentum=config.momentum,
    #           weight_decay=config.weight_decay,loss_scale=config.loss_scale)

    loss_cb = LossCallBack(data_size=train_dataset_batch_num)
    # summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)
    # cb = [loss_cb, summary_collector]
    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * train_dataset_batch_num,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.mindspore_version == 0.5:
            ckpoint_cb = ModelCheckpoint_0_5(prefix='AVA',
                                             directory=checkpoint_dir,
                                             config=ckptconfig)
        elif args_opt.mindspore_version == 0.6:
            ckpoint_cb = ModelCheckpoint_0_6(prefix='AVA',
                                             directory=checkpoint_dir,
                                             config=ckptconfig)
        cb += [ckpoint_cb]

    # model = Model(net,metrics={'knn_acc':KnnEval(batch_size=config.batch_size,device_num=1)},eval_network=eval_network)
    # model = Model(linear_layer_resnet, metrics={'accuracy': nn.Accuracy()},
    #               eval_network=linear_layer_resnet)

    model = Model(linear_layer_resnet, loss, opt, metrics={'acc': DistAccuracy(batch_size=50, device_num=device_num)},
                  eval_network=eval_network)

    model.init(train_dataset, eval_dataset)
    # model =Model(net)

    logger.info("save configs...")
    print("save configs...")
    # save current config
    config_name = 'config.json'
    save_config([os.path.join(checkpoint_dir, config_name)], config)

    logger.info("training begins...")
    print("training begins...")
    for epoch_idx in range(1, config.epochs + 1):
        ckpoint_cb.set_epoch(epoch_idx)
        model.train(1, train_dataset, callbacks=cb, dataset_sink_mode=True)
        eval_start = time.time()
        output = model.eval(eval_dataset)
        eval_cost = time.time() - eval_start
        acc = float(output["acc"])
        ckpoint_cb.set_acc(acc)
        time_cost = loss_cb.get_per_step_time()
        loss = loss_cb.get_loss()
        print("the {} epoch's resnet result: "
              " training loss {}, acc {}, "
              "training per step cost {:.2f} s, eval cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, loss, acc, time_cost, eval_cost, time_cost * train_dataset_batch_num + eval_cost))
        logger.info("the {} epoch's resnet result: "
                    " training loss {}, acc {}, "
                    "training per step cost {:.2f} s, eval cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, loss, acc, time_cost, eval_cost, time_cost * train_dataset_batch_num + eval_cost))

    if args_opt.use_moxing:
        print("download file to obs...")
        mox.file.copy_parallel(src_url=os.path.join(temp_path, config.prefix),
                               dst_url=os.path.join(config.moxing_model_save_path, config.prefix))
