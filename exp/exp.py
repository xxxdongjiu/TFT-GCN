from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from models.model5 import Model

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
        self.model_name = "wo/GCN"

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
        test_data, test_loader = self._get_data(flag="test")
        train_data, train_loader = self._get_data(flag="train")
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

        attens_energy = []

        self.model.eval()
        anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()

                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()

            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average="binary")
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        folder_path = os.path.join("./test_results/", self.model_name, self.args.data, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.args.is_pretraining:
            file_path = folder_path + "/" + "result_pretrain.txt"
        else:
            file_path = folder_path + "/" + "result.txt"
        f = open(file_path, "a")
        f.write(
            "anomaly_ratio: {:0.2f}, seq_len: {}, batch_size: {}, d_model: {}, d_ff: {}, e_layers: {}, fnet_d_model: {}, fnet_d_ff: {}, fnet_layers: {}, hidden: {}, nvechidden: {}".format(
                self.args.anomaly_ratio,
                self.args.seq_len,
                self.args.batch_size,
                self.args.d_model,
                self.args.d_ff,
                self.args.e_layers,
                self.args.fnet_d_model,
                self.args.fnet_d_ff,
                self.args.fnet_layers,
                self.args.hidden,
                self.args.nvechidden,
            )
        )
        f.write("\n")
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
