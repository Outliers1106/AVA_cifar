import mindspore.dataset as ds
from optparse import OptionParser
import os
import PIL
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import random
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from src.RandAugment import RandAugment
from src.autoaugment import CIFAR10Policy, SVHNPolicy
from src.GaussianBlur import GaussianBlur
#import mindspore.dataset.transforms.vision.py_transforms as transforms
import mindspore.dataset.vision.py_transforms as transforms
from mindspore.dataset.transforms.py_transforms import Compose
import numpy as np


# random.seed(1)
# np.random.seed(1)
# ds.config.set_seed(1)


class CIFAR10Dataset():
    def __init__(self, data_dir, training=True, use_third_trsfm=False, use_auto_augment=False, num_parallel_workers=8,
                 device_num=1, device_id="0"):

        if not training:
            trsfm = Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            if not use_third_trsfm:
                trsfm = Compose([
                    transforms.ToPIL(),
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomGrayscale(prob=0.2),
                    transforms.RandomHorizontalFlip(),
                    # GaussianBlur(kernel_size=int(0.1 * 32)),
                    # GaussianBlur(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                if use_auto_augment:
                    trsfm = Compose([
                        transforms.ToPIL(),
                        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                        transforms.RandomHorizontalFlip(),
                        CIFAR10Policy(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                else:
                    rand_augment = RandAugment(n=2, m=10)
                    trsfm = Compose([
                        transforms.ToPIL(),
                        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                        transforms.RandomHorizontalFlip(),
                        rand_augment,
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])

        self.trsfm = trsfm
        self.data_dir = data_dir
        self.num_parallel_workers = num_parallel_workers
        self.device_num = device_num
        self.device_id = device_id

    def get_dataset(self):
        print("get data from dir:", self.data_dir)
        if self.device_num == 1:
            ds_ = ds.Cifar10Dataset(self.data_dir, num_parallel_workers=self.num_parallel_workers)
        else:
            ds_ = ds.Cifar10Dataset(self.data_dir, num_parallel_workers=self.num_parallel_workers,
                                    num_shards=self.device_num, shard_id=self.device_id)

        ds_ = ds_.map(input_columns=["image"], operations=self.trsfm)
        typecast_op = C.TypeCast(mstype.int32)
        ds_ = ds_.map(input_columns="label", operations=typecast_op)
        return ds_


def makeup_train_dataset(ds1, ds2, ds3, batchsize, epoch):
    ds1 = ds1.rename(input_columns=["label", "image"], output_columns=["label1", "data1"])
    ds2 = ds2.rename(input_columns=["label", "image"], output_columns=["label2", "data2"])
    ds3 = ds3.rename(input_columns=["image"], output_columns=["data3"])
    ds_new = ds.zip((ds1, ds2))
    ds_new = ds_new.project(columns=['data1', 'data2'])
    ds_new = ds.zip((ds3, ds_new))
    ds_new = ds_new.map(input_columns=['label'], output_columns=['label'],
                        column_order=['data3', 'data2', 'data1', 'label'],
                        operations=lambda x: x)
    # to keep the order : data3 data2 data1 label

    # ds_new = ds_new.shuffle(ds_new.get_dataset_size())
    print("dataset batchsize:",batchsize)
    ds_new = ds_new.batch(batchsize)
    ds_new = ds_new.repeat(epoch)

    print("batch_size:", ds_new.get_batch_size(), "batch_num:", ds_new.get_dataset_size())

    # for data in ds_new.create_dict_iterator():
    #     print("new dataset:")
    #     print(data.keys())
    #     for key,value in data.items():
    #         print("key:",key)
    #
    #     break

    return ds_new


def makeup_test_dataset(ds_test, batchsize, epoch=1):
    ds_test = ds_test.batch(batchsize)
    ds_test = ds_test.repeat(epoch)

    return ds_test


def get_train_dataset(train_data_dir, batchsize, epoch, mode="training", device_num=1, device_id="0"):
    if mode == "linear_eval":
        cifar10_train = CIFAR10Dataset(data_dir=train_data_dir, training=False, use_third_trsfm=False)
        cifar10_train = cifar10_train.get_dataset()
        cifar10_train = makeup_test_dataset(cifar10_train, batchsize, epoch)
        return cifar10_train

    cifar10_train_1 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=False,
                                     device_num=device_num, device_id=device_id)
    cifar10_train_2 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=False,
                                     device_num=device_num, device_id=device_id)
    cifar10_train_3 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=True,
                                     use_auto_augment=False, device_num=device_num, device_id=device_id)
    cifar10_train_dataset1 = cifar10_train_1.get_dataset()
    cifar10_train_dataset2 = cifar10_train_2.get_dataset()
    cifar10_train_dataset3 = cifar10_train_3.get_dataset()
    cifar10_train_dataset = makeup_train_dataset(cifar10_train_dataset1, cifar10_train_dataset2, cifar10_train_dataset3,
                                                 batchsize=batchsize, epoch=epoch)

    return cifar10_train_dataset


def get_test_dataset(test_data_dir, batchsize, epoch=1, device_num=1, device_id="0"):
    cifar10_test = CIFAR10Dataset(data_dir=test_data_dir, training=False, use_third_trsfm=False,
                                  device_num=device_num, device_id=device_id)
    cifar10_test_dataset = cifar10_test.get_dataset()
    cifar10_test_dataset = makeup_test_dataset(cifar10_test_dataset, batchsize=batchsize, epoch=epoch)

    return cifar10_test_dataset


def get_train_test_dataset(train_data_dir, test_data_dir, batchsize, epoch=1):
    cifar10_test = CIFAR10Dataset(data_dir=test_data_dir, training=False, use_third_trsfm=False)
    cifar10_train = CIFAR10Dataset(data_dir=train_data_dir, training=False, use_third_trsfm=False)

    cifar10_test_dataset = cifar10_test.get_dataset()
    cifar10_train_dataset = cifar10_train.get_dataset()

    func0 = lambda x, y: (x, y, np.array(0, dtype=np.int32))
    func1 = lambda x, y: (x, y, np.array(1, dtype=np.int32))
    input_cols = ["image", "label"]
    output_cols = ["image", "label", "training"]
    cols_order = ["image", "label", "training"]
    cifar10_test_dataset = cifar10_test_dataset.map(input_columns=input_cols, output_columns=output_cols,
                                                    operations=func0, column_order=cols_order)
    cifar10_train_dataset = cifar10_train_dataset.map(input_columns=input_cols, output_columns=output_cols,
                                                      operations=func1, column_order=cols_order)
    # cifar10_train_dataset = cifar10_train_dataset.shuffle(cifar10_train_dataset.get_dataset_size())
    # cifar10_test_dataset = cifar10_test_dataset.shuffle(cifar10_test_dataset.get_dataset_size())
    concat_dataset = cifar10_train_dataset + cifar10_test_dataset
    concat_dataset = concat_dataset.batch(batchsize)
    concat_dataset = concat_dataset.repeat(epoch)

    return concat_dataset


# if __name__ == "__main__":
#     TRAIN_DATA_DIR = "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train"
#     train_dataset = get_train_dataset(TRAIN_DATA_DIR,128,200)
#     for data in train_dataset.create_dict_iterator():
#         print(data['data1'].shape) # (128,3,32,32)
#         print(data['data2'].shape) # (128,3,32,32)
#         print(data['data3'].shape) # (128,3,32,32)
#         print(data['label'].shape) # (128,)
#         print(data.keys())
#         break


if __name__ == "__main__":
    '''环境变量参数'''
    use_moxing = True
    if use_moxing:
        import moxing as mox

        # define local data path
        local_data_path = '/cache/data'

        mox.file.copy_parallel(src_url='s3://tuyanlun/data/', dst_url=local_data_path)
        # img = PIL.Image.open(os.path.join(local_data_path, 'GUI.png'))
        # print(img)
        TRAIN_DATA_DIR = os.path.join(local_data_path, "cifar10/cifar-10-batches-bin/train")
        TEST_DATA_DIR = os.path.join(local_data_path, "cifar10/cifar-10-batches-bin/test")
    else:
        TRAIN_DATA_DIR = '/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train'
        TEST_DATA_DIR = '/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/test'

    cifar10_train_dataset = get_train_dataset(TRAIN_DATA_DIR, 128, 200)
    get_train_test_dataset(TRAIN_DATA_DIR, TEST_DATA_DIR, 128, 1)
    for data in cifar10_train_dataset.create_dict_iterator():
        print(data.keys())
        print(data)
        break
    # print(TRAIN_DATA_DIR)
    # cifar10_train_dataset = get_train_dataset(TRAIN_DATA_DIR, 128, 200)
    #
    # def inverse_normal(data):
    #     '''
    #     Args:
    #         data: np.ndarray
    #     Returns:
    #     '''
    #     print(data.shape,type(data))
    #     mean = [0.4914, 0.4822, 0.4465]
    #     std = [0.2023, 0.1994, 0.2010]
    #     for i in range(len(mean)): # 反标准化
    #         data[i] = data[i] * std[i] + mean[i]
    #     data = data * 255
    #     data = data.astype(np.uint8)
    #     data = np.transpose(data,(1,2,0)) # (ch,h,w) -> (h,w,ch)
    #     return data
    #
    # def plot_img(data):
    #     '''
    #
    #     Args:
    #         data: np.ndarray
    #
    #     '''
    #     data = Image.fromarray(data,mode='RGB')
    #     print(data)
    #     #data.show()
    #     data.save('test.jpg')
    #
    #
    # for data in cifar10_train_dataset.create_dict_iterator():
    #     print(data['data1'].shape)
    #     t=inverse_normal(data['data1'][0])
    #     print(type(t),t)
    #     plot_img(t)
    #     #print(data['data2'])
    #     #print(data['data3'])
    #     print(data['label'])
    #     break
