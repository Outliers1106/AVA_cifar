import time
import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class WithLossCell(nn.Cell):
    """
        Wrap the network with loss function to compute loss.

        Args:
            backbone (Cell): The target network to wrap.
            loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data3, data2, data1, label):
        feature3, feature2, feature1 = self._backbone(data3, data2, data1)
        return self._loss_fn(feature3, feature2, feature1, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
        Network training package class.

        Append an optimizer to the training network after that the construct function
        can be called to create the backward graph.

        Args:
            net_with_loss (Cell): The training network with loss.
            optimizer (Cell): Optimizer for updating the weights.
            sens (Number): The adjust parameter. Default value is 1.0.
            reduce_flag (bool): The reduce flag. Default value is False.
            mean (bool): Allreduce method. Default value is False.
            degree (int): Device number. Default value is None.
    """

    def __init__(self, net_with_loss, optimizer, sens=1.0, reduce_flag=False, mean=False, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.net_with_loss = net_with_loss
        self.weights = ParameterTuple(net_with_loss.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=False)
        # self.sens = Tensor((np.ones((1,)) * sens).astype(np.float32))
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, data3, data2, data1, label):
        weights = self.weights
        loss = self.net_with_loss(data3, data2, data1, label)
        grads = self.grad(self.net_with_loss, weights)(data3, data2, data1, label)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


if __name__ == "__main__":
    from mindspore import context

    # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


    class test(nn.Cell):
        def __init__(self):
            super(test, self).__init__()
            self.eye_matrix = P.Eye()(4, 4, mindspore.float32)
            self.reduce_sum = P.ReduceSum(keep_dims=True)
            self.sum = P.ReduceSum()
            self.sum_keep_dim = P.ReduceSum(keep_dims=True)

        def construct(self, x):
            # return P.ReduceSum()(x, 0)
            print("l2 norm", P.L2Normalize(axis=1)(x))
            print("sum_keep_dim(x,1)", self.sum_keep_dim(x, 1), "shape", P.Shape()(self.sum_keep_dim(x, 1)))
            print("sum_keep_dim(x,0)", self.sum_keep_dim(x, 0), "shape", P.Shape()(self.sum_keep_dim(x, 0)))
            print("sum(x,1)", self.sum_keep_dim(x, 1), "shape", P.Shape()(self.sum(x, 1)))
            return self.sum_keep_dim(x, 1) / self.sum_keep_dim(x, 1)
            matrix = x * self.eye_matrix
            return self.reduce_sum(matrix, 0)
            return P.DiagPart()(x)
            # return P.ReduceMean()(x)
            # perm = (1, 0)
            # return P.Transpose()(x, perm)

            myprint = P.Print()
            shape = P.Shape()
            myprint("test")

            # split = P.Split(0, 3)
            # output = split(x)
            # return output


    t_input = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))
    net = test()
    res1 = net(t_input)
    print("the result:", res1)

    # feature3 = Tensor(np.random.rand(100, 128).astype(np.float32))
    # feature2 = Tensor(np.random.rand(100, 128).astype(np.float32))
    # feature1 = Tensor(np.random.rand(100, 128).astype(np.float32))
    # #
    # losses = LossNet()
    # # print('test')
    # loss = losses(feature3, feature2, feature1, feature1)
    # print("loss", loss)
