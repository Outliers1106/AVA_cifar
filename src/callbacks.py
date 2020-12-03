import time
import os
import shutil
import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
import threading
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
import mindspore.context as context


class LossCallBack(Callback):
    """
        Monitor the loss in training.

        If the loss is NAN or INF terminating training.

        Note:
            If per_print_times is 0 do not print loss.

        Args:
            per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, data_size, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0
        self.data_size = data_size
        self.step_cnt = 0
        self.loss_sum = 0

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = time.time() - self.epoch_time
        self._per_step_seconds = epoch_seconds / self.data_size
        self._loss = self.loss_sum / self.step_cnt
        self.step_cnt = 0
        self.loss_sum = 0

    def get_loss(self):
        return self._loss

    def get_per_step_time(self):
        return self._per_step_seconds

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if not isinstance(cb_params.net_outputs, list):
            loss = cb_params.net_outputs.asnumpy()
        else:
            loss = cb_params.net_outputs[0].asnumpy()

        # cb_params.batch_num means : dataset_size / batch_size
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        # print("cur_step_in_epoch:",cur_step_in_epoch,"step loss:",loss)
        # if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
        self.loss_sum += loss
        self.step_cnt += 1
        # print("check speed step{}".format(cur_step_in_epoch))


class LossCallBack_imagenet(Callback):
    """
        Monitor the loss in training.

        If the loss is NAN or INF terminating training.

        Note:
            If per_print_times is 0 do not print loss.

        Args:
            per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, data_size, per_print_times=1, logger=None):
        super(LossCallBack_imagenet, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.logger = logger
        self._per_print_times = per_print_times
        self._loss = 0
        self.data_size = data_size
        self.step_cnt = 0
        self.loss_sum = 0

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = time.time() - self.epoch_time
        self._per_step_seconds = epoch_seconds / self.data_size
        self._loss = self.loss_sum / self.step_cnt
        self.step_cnt = 0
        self.loss_sum = 0

        cb_params = run_context.original_args()
        epoch_idx = (cb_params.cur_step_num - 1) // cb_params.batch_num + 1

        self.logger.info("the {} epoch's resnet result: "
                         " training loss {},"
                         "training per step cost {:.2f} s, total_cost {:.2f} s".format(
            epoch_idx, self._loss, self._per_step_seconds, self._per_step_seconds * cb_params.batch_num))

    def get_loss(self):
        return self._loss

    def get_per_step_time(self):
        return self._per_step_seconds

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if not isinstance(cb_params.net_outputs, list):
            loss = cb_params.net_outputs.asnumpy()
        else:
            loss = cb_params.net_outputs[0].asnumpy()

        # cb_params.batch_num means : dataset_size / batch_size
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        # print("cur_step_in_epoch:",cur_step_in_epoch,"step loss:",loss)
        # if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
        self.loss_sum += loss
        self.step_cnt += 1
        # print("check speed step{}".format(cur_step_in_epoch))



class ModelCheckpoint_(ModelCheckpoint):
    def __init__(self, prefix='CKP', directory=None, config=None, epoch_all=200, monitor="acc", mode="max"):
        super(ModelCheckpoint_, self).__init__(prefix, directory, config)
        self.prefix_origin = prefix
        self.force_to_save = False
        self.acc = 0
        self.epoch = 0

    def step_end(self, run_context):
        pass
    
    def epoch_begin(self, run_context):
        if self.force_to_save:
            self.prefix = self.prefix_origin + "-" + str(self.acc)
            cb_params = run_context.original_args()
            cb_params.cur_step_num = self.epoch
            self._save_ckpt(cb_params=cb_params,force_to_save=True)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_force_to_save(self, flag):
        self.force_to_save = flag

    def set_acc(self, acc):
        self.acc = acc
