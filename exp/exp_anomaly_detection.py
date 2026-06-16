from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    @staticmethod
    def _bytes_to_mb(value):
        return float(value) / (1024.0 * 1024.0)

    def _sync_device(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.synchronize()

    def _profile_inference_resources(self, sample_batch):
        model_for_stats = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        total_params = sum(p.numel() for p in model_for_stats.parameters())
        trainable_params = sum(p.numel() for p in model_for_stats.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in model_for_stats.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model_for_stats.buffers())
        model_weight_mb = self._bytes_to_mb(param_bytes + buffer_bytes)

        warmup_steps = max(0, int(getattr(self.args, 'infer_warmup_steps', 10)))
        repeat_steps = max(1, int(getattr(self.args, 'infer_repeat_steps', 50)))
        f_dim = -1 if self.args.features == 'MS' else 0

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for _ in range(warmup_steps):
                warmup_out = self.model(sample_batch, None, None, None)
                if isinstance(warmup_out, tuple) and len(warmup_out) == 3:
                    warmup_out = warmup_out[0]
                elif isinstance(warmup_out, tuple):
                    warmup_out = warmup_out[0]
                _ = warmup_out[:, :, f_dim:]

            timings = []
            for _ in range(repeat_steps):
                self._sync_device()
                tic = time.perf_counter()
                infer_out = self.model(sample_batch, None, None, None)
                if isinstance(infer_out, tuple) and len(infer_out) == 3:
                    infer_out = infer_out[0]
                elif isinstance(infer_out, tuple):
                    infer_out = infer_out[0]
                _ = infer_out[:, :, f_dim:]
                self._sync_device()
                timings.append(time.perf_counter() - tic)

        metrics = {
            'device_type': self.device.type,
            'batch_size': int(sample_batch.shape[0]),
            'seq_len': int(sample_batch.shape[1]),
            'feature_dim': int(sample_batch.shape[2]),
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'model_weight_mb': float(model_weight_mb),
            'input_batch_mb': float(self._bytes_to_mb(sample_batch.numel() * sample_batch.element_size())),
            'infer_warmup_steps': int(warmup_steps),
            'infer_repeat_steps': int(repeat_steps),
        }

        return metrics

    def _get_model_output_module(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _maybe_inverse_transform(self, dataset, array):
        if dataset is None or not hasattr(dataset, 'inverse_transform'):
            return array
        try:
            return dataset.inverse_transform(array)
        except Exception:
            return array

    def train(self, setting):
        train_data, original_train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        if not hasattr(train_data, 'train_folds_x'):
            folds_x = [train_data.train_windows]
        else:
            folds_x = train_data.train_folds_x

        folds_y = [np.zeros((fx.shape[0],) + test_data.test_label_windows.shape[1:]) for fx in folds_x]

        num_folds = len(folds_x)

        for cv_fold in range(num_folds):
            print(f"====== Cross Validation Fold {cv_fold + 1}/{num_folds} ======")

            self.model = self._build_model().to(self.device)
            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            if num_folds > 1:
                cur_train_x = np.concatenate([folds_x[i] for i in range(num_folds) if i != cv_fold], axis=0)
                cur_train_y = np.concatenate([folds_y[i] for i in range(num_folds) if i != cv_fold], axis=0)
                train_dataset = torch.utils.data.TensorDataset(torch.tensor(cur_train_x), torch.tensor(cur_train_y))
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
            else:
                train_loader = original_train_loader

            path = os.path.join(self.args.checkpoints, setting, f"fold_{cv_fold}")
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()
            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            mid_epoch = self.args.train_epochs // 2
            phase1_completed = False

            for epoch in range(self.args.train_epochs):
                model_name = getattr(self.args, 'model', '')
                actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                CASE_1_MODELS = ['Ours_MemAE', 'Ours_VAE', 'Ours_DeepKoopman']
                CASE_2_MODELS = ['Ours_Koopman_MemAE', 'Ours_Koopman_VAE']
                CASE_3_MODELS = ['Ours', 'Ours_Koopman_double_MemAE', 'Ours_Koopman_double_VAE']

                is_stage1 = False

                if model_name in CASE_2_MODELS + CASE_3_MODELS:
                    is_stage1 = epoch < mid_epoch

                if epoch == mid_epoch and not phase1_completed:
                    branch_type = 'Dual branch' if model_name in CASE_3_MODELS else 'Single branch'
                    print(f"------ Switching to Phase 2: Training {branch_type} ------")
                    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
                    phase1_completed = True

                if getattr(actual_model, 'koopman', None) is not None:
                    for param in actual_model.koopman.parameters(): param.requires_grad = is_stage1
                if getattr(actual_model, 'mra_rec', None) is not None:
                    for param in actual_model.mra_rec.parameters(): param.requires_grad = not is_stage1
                if getattr(actual_model, 'wavelet_vae', None) is not None:
                    for param in actual_model.wavelet_vae.parameters(): param.requires_grad = not is_stage1
                if getattr(actual_model, 'output_projection', None) is not None:
                    for param in actual_model.output_projection.parameters(): param.requires_grad = not is_stage1

                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x, None, None, None)
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        outputs, error_time, error_freq = outputs
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]

                    if getattr(actual_model, 'control_dim', 0) > 0:
                        state_true = batch_x[:, :, actual_model.control_dim:]
                    else:
                        state_true = batch_x

                    loss = 0.0

                    if is_stage1:
                        loss_koopman_dyn = criterion(actual_model.latest_pred_z, actual_model.latest_z.detach())
                        loss_koopman_rec = criterion(actual_model.latest_state_rec, state_true)
                        lambda_dyn, lambda_rec = 0.5, 0.5
                        loss = lambda_rec * loss_koopman_rec + lambda_dyn * loss_koopman_dyn
                    elif model_name in CASE_3_MODELS:
                        loss_recon = criterion(outputs, batch_x)
                        loss = 1.0 * loss_recon

                        if hasattr(actual_model, 'latest_vq_loss') and actual_model.latest_vq_loss is not None:
                            loss += 0.1 * actual_model.latest_vq_loss

                        if hasattr(actual_model, 'latest_dwt_P') and actual_model.latest_dwt_P is not None:
                            dwt_P = actual_model.latest_dwt_P
                            recon_dwt_P = actual_model.latest_recon_dwt_P
                            mu_P = actual_model.latest_mu_P
                            logvar_P = actual_model.latest_logvar_P
                            loss_vae_recon = criterion(recon_dwt_P, dwt_P)
                            loss_vae_kl = -0.5 * torch.mean(1 + logvar_P - mu_P.pow(2) - logvar_P.exp())
                            loss += 0.5 * loss_vae_recon + 0.05 * loss_vae_kl
                    elif model_name in CASE_2_MODELS:
                        loss_recon = criterion(outputs, batch_x)
                        if model_name == 'Ours_Koopman_VAE':
                            dwt_P = actual_model.latest_dwt_P
                            recon_dwt_P = actual_model.latest_recon_dwt_P
                            mu_P = actual_model.latest_mu_P
                            logvar_P = actual_model.latest_logvar_P
                            loss_vae_recon = criterion(recon_dwt_P, dwt_P)
                            loss_vae_kl = -0.5 * torch.mean(1 + logvar_P - mu_P.pow(2) - logvar_P.exp())
                            loss_vae_total = 0.5 * loss_vae_recon + 0.05 * loss_vae_kl
                            loss = 1.0 * loss_recon + loss_vae_total
                        else:
                            loss = loss_recon
                    elif model_name in CASE_1_MODELS:
                        if model_name == 'Ours_DeepKoopman':
                            loss_recon = criterion(outputs, batch_x)
                            loss_koopman_dyn = criterion(actual_model.latest_pred_z, actual_model.latest_z.detach())
                            loss_koopman_rec = criterion(actual_model.latest_state_rec, state_true)
                            loss = loss_recon + 0.5 * loss_koopman_dyn + 0.5 * loss_koopman_rec
                        elif model_name == 'Ours_VAE':
                            dwt_P = actual_model.latest_dwt_P
                            recon_dwt_P = actual_model.latest_recon_dwt_P
                            mu_P = actual_model.latest_mu_P
                            logvar_P = actual_model.latest_logvar_P
                            loss_vae_recon = criterion(recon_dwt_P, dwt_P)
                            loss_vae_kl = -0.5 * torch.mean(1 + logvar_P - mu_P.pow(2) - logvar_P.exp())
                            loss = 0.5 * loss_vae_recon + 0.05 * loss_vae_kl
                        elif model_name == 'Ours_MemAE':
                            loss = criterion(outputs, batch_x)
                    else:
                        loss = criterion(outputs, batch_x)

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            should_apply_early_stopping = True
            if model_name in CASE_2_MODELS + CASE_3_MODELS:
                should_apply_early_stopping = epoch >= mid_epoch

            if should_apply_early_stopping:
                early_stopping(train_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, original_test_loader = self._get_data(flag='test')
        train_data, original_train_loader = self._get_data(flag='train')

        if not hasattr(train_data, 'train_folds_x'):
            folds_x = [train_data.train_windows]
        else:
            folds_x = train_data.train_folds_x

        folds_y = [np.zeros((fx.shape[0],) + test_data.test_label_windows.shape[1:]) for fx in folds_x]

        num_folds = len(folds_x)

        metrics_all_folds = []
        roc_data_folds = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for cv_fold in range(num_folds):
            print(f"====== Testing CV Fold {cv_fold + 1}/{num_folds} ======")

            path = os.path.join(self.args.checkpoints, setting, f"fold_{cv_fold}", "checkpoint.pth")
            if test or num_folds > 1:
                print('loading model from:', path)
                self.model.load_state_dict(torch.load(path))

            if num_folds > 1:
                cur_val_x = folds_x[cv_fold]
                cur_val_y = folds_y[cv_fold]
                test_x = np.concatenate([test_data.test_windows, cur_val_x], axis=0)
                test_y = np.concatenate([test_data.test_label_windows, cur_val_y], axis=0)
                test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

                cur_train_x = np.concatenate([folds_x[i] for i in range(num_folds) if i != cv_fold], axis=0)
                cur_train_y = np.concatenate([folds_y[i] for i in range(num_folds) if i != cv_fold], axis=0)
                train_dataset = torch.utils.data.TensorDataset(torch.tensor(cur_train_x), torch.tensor(cur_train_y))
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            else:
                test_loader = original_test_loader
                train_loader = original_train_loader

            self._plot_dataset = test_data

            attens_energy = []
            vae_energy_list = []

            attens_energy_time = []
            attens_energy_freq = []

            self.model.eval()
            self.anomaly_criterion = nn.MSELoss(reduce=False)

            model_name = getattr(self.args, 'model', '')
            CASE_1_MODELS = ['Ours_MemAE', 'Ours_VAE', 'Ours_DeepKoopman']
            CASE_2_MODELS = ['Ours_Koopman_MemAE', 'Ours_Koopman_VAE']
            CASE_3_MODELS = ['Ours', 'Ours_Koopman_double_MemAE', 'Ours_Koopman_double_VAE']

            is_dual_branch = model_name in CASE_3_MODELS

            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x, None, None, None)

                    if is_dual_branch and isinstance(outputs, tuple) and len(outputs) == 3:
                        _, error_time, error_freq = outputs
                        score1 = error_time.detach().cpu().numpy()
                        score2 = error_freq.detach().cpu().numpy()
                        attens_energy_time.append(score1)
                        attens_energy_freq.append(score2)
                    else:
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                        score = score.detach().cpu().numpy()
                        attens_energy.append(score)

            if is_dual_branch:
                train_energy_time = np.concatenate(attens_energy_time, axis=0).reshape(-1)
                train_energy_freq = np.concatenate(attens_energy_freq, axis=0).reshape(-1)
            else:
                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
                train_energy = np.array(attens_energy)

            attens_energy = []
            test_energy_time_list = []
            test_energy_freq_list = []
            test_labels = []

            X_collect, M_collect, P_collect, Y_collect = [], [], [], []
            actual_model = self._get_model_output_module()

            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                if getattr(self.args, 'model', '') in ['Ours']:
                    latest_M = getattr(actual_model, 'latest_M', None)
                    latest_P = getattr(actual_model, 'latest_P', None)
                    if latest_M is not None and latest_P is not None:
                        x_c = self._maybe_inverse_transform(self._plot_dataset, batch_x.detach().cpu().numpy())
                        m_c = self._maybe_inverse_transform(self._plot_dataset, latest_M.detach().cpu().numpy())
                        p_c = self._maybe_inverse_transform(self._plot_dataset, latest_P.detach().cpu().numpy())
                        X_collect.append(x_c)
                        M_collect.append(m_c)
                        P_collect.append(p_c)
                        Y_collect.append(batch_y.detach().cpu().numpy())

                if is_dual_branch and isinstance(outputs, tuple) and len(outputs) == 3:
                    _, error_time, error_freq = outputs
                    score1 = error_time.detach().cpu().numpy()
                    score2 = error_freq.detach().cpu().numpy()
                    test_energy_time_list.append(score1)
                    test_energy_freq_list.append(score2)
                else:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)

                test_labels.append(batch_y.detach().cpu().numpy())

            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            gt = test_labels.astype(int)

            if is_dual_branch:
                test_energy_time = np.concatenate(test_energy_time_list, axis=0).reshape(-1)
                test_energy_freq = np.concatenate(test_energy_freq_list, axis=0).reshape(-1)

                print("Searching for optimal thresholds (Time & Freq) in 0% - 25% anomaly ratio based on Raw F1...")
                best_f1_raw = -1
                best_threshold_time = -1
                best_threshold_freq = -1
                best_ratio = (-1, -1)

                for ratio_time in np.arange(0.1, 1.1, 1):
                    for ratio_freq in np.arange(0.1, 1.1, 1):
                        temp_thresh_time = np.percentile(train_energy_time, 100 - ratio_time)
                        temp_thresh_freq = np.percentile(train_energy_freq, 100 - ratio_freq)

                        temp_pred = ((test_energy_time > temp_thresh_time) | (test_energy_freq > temp_thresh_freq)).astype(int)
                        _, _, f_score_raw_temp, _ = precision_recall_fscore_support(
                            gt, temp_pred, average='binary', zero_division=0)

                        if f_score_raw_temp > best_f1_raw:
                            best_f1_raw = f_score_raw_temp
                            best_threshold_time = temp_thresh_time
                            best_threshold_freq = temp_thresh_freq
                            best_ratio = (ratio_time, ratio_freq)

                threshold_time = best_threshold_time
                threshold_freq = best_threshold_freq
                print("Optimal Anomaly Ratio : Time {:.2f}%, Freq {:.2f}%".format(best_ratio[0], best_ratio[1]))
                print("Optimal Threshold Time :", threshold_time)
                print("Optimal Threshold Freq :", threshold_freq)

                pred_raw = ((test_energy_time > threshold_time) | (test_energy_freq > threshold_freq)).astype(int)

                denom_time = np.mean(train_energy_time) if np.mean(train_energy_time) > 1e-8 else 1e-8
                denom_freq = np.mean(train_energy_freq) if np.mean(train_energy_freq) > 1e-8 else 1e-8
                combined_energy = (test_energy_time / denom_time) + (test_energy_freq / denom_freq)
                auc = roc_auc_score(gt, combined_energy)

                test_energy = combined_energy
                threshold = 2.0
            else:
                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
                test_energy = np.array(attens_energy)

                print("Searching for optimal threshold in 0% - 25% anomaly ratio based on Raw F1...")
                best_f1_raw = -1
                best_threshold = -1
                best_ratio = -1

                for ratio in np.arange(0.1, 25.1, 1):
                    temp_threshold = np.percentile(train_energy, 100 - ratio)
                    temp_pred = (test_energy > temp_threshold).astype(int)
                    _, _, f_score_raw_temp, _ = precision_recall_fscore_support(
                        gt, temp_pred, average='binary', zero_division=0)

                    if f_score_raw_temp > best_f1_raw:
                        best_f1_raw = f_score_raw_temp
                        best_threshold = temp_threshold
                        best_ratio = ratio

                threshold = best_threshold
                print("Optimal Anomaly Ratio : {:.2f}%".format(best_ratio))
                print("Optimal Threshold :", threshold)

                pred_raw = (test_energy > threshold).astype(int)
                auc = roc_auc_score(gt, test_energy)

            fpr, tpr, _ = roc_curve(gt, test_energy)

            gt_diff = np.diff(np.concatenate(([0], gt, [0])))
            anomaly_starts = np.where(gt_diff == 1)[0]
            anomaly_ends = np.where(gt_diff == -1)[0]

            ttd_list = []
            missed_segments = 0
            total_segments = len(anomaly_starts)

            for s, e in zip(anomaly_starts, anomaly_ends):
                seg_pred = pred_raw[s:e]

                detected = np.where(seg_pred == 1)[0]
                if len(detected) > 0:
                    ttd_list.append(detected[0])
                else:
                    missed_segments += 1

            ttd_raw = np.mean(ttd_list) if len(ttd_list) > 0 else 0.0
            mr_raw = missed_segments / total_segments if total_segments > 0 else 0.0

            accuracy_raw = accuracy_score(gt, pred_raw)
            precision_raw, recall_raw, f_score_raw, _ = precision_recall_fscore_support(
                gt, pred_raw, average='binary', zero_division=0)

            print(f"Fold {cv_fold + 1} Raw metrics:")
            print("  Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, TTD : {:0.4f}, MR : {:0.4f}".format(
                accuracy_raw, precision_raw, recall_raw, f_score_raw, ttd_raw, mr_raw))
            print("  AUC      : {:0.4f}".format(auc))

            gt_adj, pred_adj = adjustment(gt.copy(), pred_raw.copy())
            accuracy_adj = accuracy_score(gt_adj, pred_adj)
            precision_adj, recall_adj, f_score_adj, _ = precision_recall_fscore_support(
                gt_adj, pred_adj, average='binary', zero_division=0)

            print(f"Fold {cv_fold + 1} Adjusted metrics:")
            print("  Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
                accuracy_adj, precision_adj, recall_adj, f_score_adj, auc))

            metrics_all_folds.append({
                'raw': (precision_raw, recall_raw, f_score_raw, auc, ttd_raw, mr_raw),
                'adj': (precision_adj, recall_adj, f_score_adj, auc)
            })

        if cv_fold == num_folds - 1:
            raw_precs = [m['raw'][0] for m in metrics_all_folds]
            raw_recs = [m['raw'][1] for m in metrics_all_folds]
            raw_f1s = [m['raw'][2] for m in metrics_all_folds]
            raw_aucs = [m['raw'][3] for m in metrics_all_folds]
            raw_ttds = [m['raw'][4] for m in metrics_all_folds]
            raw_mrs = [m['raw'][5] for m in metrics_all_folds]

            adj_precs = [m['adj'][0] for m in metrics_all_folds]
            adj_recs = [m['adj'][1] for m in metrics_all_folds]
            adj_f1s = [m['adj'][2] for m in metrics_all_folds]
            adj_aucs = [m['adj'][3] for m in metrics_all_folds]

            with open("result_anomaly_detection_k_fold.txt", 'a') as f:
                f.write(setting + f" ({num_folds}-Fold CV)\n")

                f.write(f"Raw CV Mean: Prec={np.mean(raw_precs):.4f} ± {np.std(raw_precs):.4f}, "
                        f"Rec={np.mean(raw_recs):.4f} ± {np.std(raw_recs):.4f}, "
                        f"F1={np.mean(raw_f1s):.4f} ± {np.std(raw_f1s):.4f}, "
                        f"AUC={np.mean(raw_aucs):.4f} ± {np.std(raw_aucs):.4f}, "
                        f"TTD={np.mean(raw_ttds):.4f} ± {np.std(raw_ttds):.4f}, "
                        f"MR={np.mean(raw_mrs):.4f} ± {np.std(raw_mrs):.4f}\n")

                f.write(f"Adj CV Mean: Prec={np.mean(adj_precs):.4f} ± {np.std(adj_precs):.4f}, "
                        f"Rec={np.mean(adj_recs):.4f} ± {np.std(adj_recs):.4f}, "
                        f"F1={np.mean(adj_f1s):.4f} ± {np.std(adj_f1s):.4f}, "
                        f"AUC={np.mean(adj_aucs):.4f} ± {np.std(adj_aucs):.4f}\n")
                f.write('\n')

        return
