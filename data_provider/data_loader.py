import os
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class UAVSegLoader(Dataset):

    def __init__(self, args, root_path, win_size, step=10, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_files = sorted(glob.glob(os.path.join(root_path, 'train', '*.csv')))
        test_files = sorted(glob.glob(os.path.join(root_path, 'test', '*.csv')))
        label_files = sorted(glob.glob(os.path.join(root_path, 'test_label', '*.csv')))

        train_segments = []
        for f in train_files:
            df = pd.read_csv(f)
            data = np.nan_to_num(df.values[:, 1:])
            train_segments.append(data)

        train_concat = np.concatenate(train_segments, axis=0)
        self.scaler.fit(train_concat)

        train_segments = [self.scaler.transform(seg) for seg in train_segments]

        np.random.seed(42)
        indices = np.arange(len(train_segments))
        np.random.shuffle(indices)
        fold_indices = np.array_split(indices, 5)

        self.train_folds_x = []
        self.train_folds_y = []
        for fold in fold_indices:
            fold_x = []
            for idx in fold:
                seg = train_segments[idx]
                n = len(seg)
                for i in range(0, n - win_size + 1, self.step):
                    fold_x.append(seg[i:i + win_size])
            if len(fold_x) > 0:
                fold_x = np.array(fold_x)
                fold_y = np.zeros_like(fold_x)
            else:
                fold_x = np.empty((0, win_size, train_segments[0].shape[1]))
                fold_y = np.empty((0, win_size, train_segments[0].shape[1]))
            self.train_folds_x.append(np.float32(fold_x))
            self.train_folds_y.append(np.float32(fold_y))

        self.train_windows = np.concatenate(self.train_folds_x, axis=0)

        train_len = len(self.train_windows)
        indices = np.arange(train_len)
        np.random.shuffle(indices)

        val_size = int(train_len * 0.00)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        self.val_windows = self.train_windows[val_idx]
        self.train_windows = self.train_windows[train_idx]

        test_segments = []
        for f in test_files:
            df = pd.read_csv(f)
            data = np.nan_to_num(df.values[:, 1:])
            data = self.scaler.transform(data)
            test_segments.append(data)

        label_map = {
            os.path.basename(f).replace('_label.csv', ''): f
            for f in label_files
        }
        label_segments = []
        for f in test_files:
            test_name = os.path.basename(f).replace('.csv', '')
            label_path = label_map.get(test_name)
            if label_path is None:
                raise FileNotFoundError(
                    f"Label file for test file '{test_name}.csv' not found in '{root_path}/test_label'.")
            df = pd.read_csv(label_path)
            lab = df.values[:, 1:]
            label_segments.append(lab)

        self.test_windows = []
        self.test_label_windows = []
        for data, lab in zip(test_segments, label_segments):
            n = len(data)
            for i in range(0, n - win_size + 1, self.step):
                self.test_windows.append(data[i:i + win_size])
                self.test_label_windows.append(lab[i:i + win_size])

        self.test_windows = np.array(self.test_windows)
        self.test_label_windows = np.array(self.test_label_windows)

        test_labels_max = np.max(self.test_label_windows, axis=(1, 2))
        normal_windows_count = np.sum(test_labels_max == 0)
        anomaly_windows_count = np.sum(test_labels_max > 0)

        print(
            f"train_cases={len(train_files)}, "
            f"test_cases={len(test_files)}, windows_train={len(self.train_windows)}, "
            f"windows_test={len(self.test_windows)}\n"
            f"  - test_normal_windows={normal_windows_count}, test_anomaly_windows={anomaly_windows_count}"
        )

    def __len__(self):
        if self.flag == "train":
            return len(self.train_windows)
        elif self.flag == "val":
            return len(self.val_windows)
        elif self.flag == "test":
            return len(self.test_windows)
        else:
            return len(self.test_windows)

    def __getitem__(self, index):
        if self.flag == "train":
            x = self.train_windows[index]
            y = np.zeros_like(x)
        elif self.flag == "val":
            x = self.val_windows[index]
            y = np.zeros_like(x)
        elif self.flag == "test":
            x = self.test_windows[index]
            y = self.test_label_windows[index]
        else:
            x = self.test_windows[index]
            y = self.test_label_windows[index]
        return np.float32(x), np.float32(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
