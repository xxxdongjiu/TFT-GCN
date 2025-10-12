from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from models.model4 import Model

torch.multiprocessing.set_sharing_strategy("file_system")
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class Exp_Anomaly_Detection(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.model_name = "review"

    def _build_model(self):
        model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):
        train_data, train_loader = self._get_data(flag="train")
        model_optim = self._select_optimizer()
        for epoch in range(self.args.pretrain_epochs):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                outputs = self.model.pretrain(batch_x)

                loss = (outputs - batch_x) ** 2
                loss = (loss * self.model.mask).sum() / self.model.mask.sum()

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                loss = criterion(pred, true)

                fft1 = torch.fft.fft(outputs.transpose(1, 2), norm="forward").detach().cpu()
                fft2 = torch.fft.fft(batch_x.transpose(1, 2), norm="forward").detach().cpu()
                fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
                fourier_loss = criterion(torch.real(fft1), torch.real(fft2)) + criterion(
                    torch.imag(fft1), torch.imag(fft2)
                )
                # fourier_loss = torch.mean(torch.abs(fft1 - fft2) ** 2)
                loss += fourier_loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        # test_data, test_loader = self._get_data(flag='test')

        if self.args.is_pretraining:
            path = os.path.join(self.args.checkpoints_pretrain, self.model_name, self.args.data, setting)
        else:
            path = os.path.join(self.args.checkpoints, self.model_name, self.args.data, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # training
        print("training ...")
        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)

                fft1 = torch.fft.fft(outputs.transpose(1, 2), norm="forward")
                fft2 = torch.fft.fft(batch_x.transpose(1, 2), norm="forward")
                fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
                fourier_loss = criterion(torch.real(fft1), torch.real(fft2)) + criterion(
                    torch.imag(fft1), torch.imag(fft2)
                )
                # fourier_loss = torch.mean(torch.abs(fft1 - fft2) ** 2)
                loss += fourier_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            # test_loss = self.vali(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            print()

        print("Total cost time: {}".format(time.time() - time_now))
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="")
        if test:
            print("loading model")
            if self.args.is_pretraining:
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(
                            "./checkpoints_pretrain/",
                            self.model_name,
                            self.args.data,
                            setting,
                            "checkpoint.pth",
                        )
                    )
                )
            else:
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(
                            "./checkpoints/",
                            self.model_name,
                            self.args.data,
                            setting,
                            "checkpoint.pth",
                        )
                    )
                )
        all_outputs = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)

            _, _, v = outputs.size()
            outputs = outputs.reshape(-1, v)
            # 转为 NumPy 并添加到列表
            outputs_np = outputs.cpu().detach().numpy()  # 移到 CPU 并转为 NumPy
            all_outputs.append(outputs_np)

        # 拼接所有批次的 outputs，形状变为 [total_time, c]
        all_outputs = np.concatenate(all_outputs, axis=0)  # 按 batch 维度拼接

        # 保存为 .npy 文件
        folder_path = os.path.join("./reconstructed_data/", self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = folder_path + "/" + "reconstructed_outputs.npy"
        np.save(save_path, all_outputs)
        print(f"Outputs saved to {save_path}")
