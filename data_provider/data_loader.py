import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


import warnings

warnings.filterwarnings("ignore")


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data) # 处理NaN值，将其替换成0
        self.scaler.fit(data) # 计算均值和标准差
        data = self.scaler.transform(data) # 将训练数据标准化
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data) 
        self.test = self.scaler.transform(test_data) 
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = pd.read_csv(os.path.join(root_path, "test_label.csv")).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )
        
class HAISegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "train1.csv"), sep=';')
        test_data = pd.read_csv(os.path.join(root_path, "test1.csv"), sep=';')
        labels = test_data.values[:, -4]
        train_data = train_data.values[:, 1:-4]
        test_data = test_data.values[:, 1:-4]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class SKABSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "train.csv"), sep=';')
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"), sep=';')
        labels = test_data.values[:, -2]
        train_data = train_data.values[:, 1:-2]
        test_data = test_data.values[:, 1:-2]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )


class CCFDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        labels = test_data.values[:, -1]
        train_data = train_data.values[:, 1:-1]
        test_data = test_data.values[:, 1:-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
            ), np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size + self.win_size
                ]
            )