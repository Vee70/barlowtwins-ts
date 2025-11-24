import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

### Local scripts
from loss import BarlowTwinsLoss
from augmentations import crop, generate_mask_wavelet_args, mask_wavelet, select_sub_series
from utils import TSDataset


class TSReprLearner:
    def __init__(
        self,
        encoder,
        proj_head,
        lr=0.003,
        lambda_coeff=1e-3,
        device=torch.device("cpu"),
    ):

        self.encoder = encoder.to(device)
        self.proj_head = proj_head.to(device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()),
        )

        self.lr = lr
        self.lambda_coeff = lambda_coeff
        self.device = device

    def maxpool(self, x):
        """ Apply max_pool over time dimension
            Input:  (N, L, C_in)
            Output: (N, C_out)
        """
        return F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)

    def encode(self, data_npy, batch_size=1):
        """ Input:  (N, L, C_in)
            Output: (N, C_out)
        """
        data_loader = DataLoader(
            TSDataset(data_npy),
            batch_size=batch_size,
        )
        self.encoder.eval()
        with torch.no_grad():
            output = [ self.maxpool(self.encoder(x.to(self.device)))
                          for x in data_loader ]
        return torch.cat(output, dim=0).cpu().numpy()

    def pretrain(
        self,
        train_data_npy,
        n_epochs,
        batch_size,
        l_min,
        l_max,
        mask_prob,
        max_dec_lv,
        wavelets,
    ):
        ### Barlow Twins Loss
        self.loss_fn = BarlowTwinsLoss(
            batch_size,
            lambda_coeff=self.lambda_coeff,
        )

        ### shape: (N, L, C)
        train_loader = DataLoader(
            TSDataset(train_data_npy),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        n_samples, seq_len, _ = train_data_npy.shape

        ### learning rate scheduler
        n_iters_per_epoch = int(np.ceil(n_samples / batch_size))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=n_epochs,
            steps_per_epoch=n_iters_per_epoch,
            three_phase=False,
        )

        ### params for `generate_mask_wavelet_args`
        dec_lvs = {
            wavelets[0]: pywt.dwt_max_level(seq_len, wavelets[0]),
            wavelets[1]: pywt.dwt_max_level(seq_len, wavelets[1]),
        }

        ### training loop
        for epoch in range(n_epochs):

            for x_batch in train_loader:

                x_batch = x_batch.to(self.device)
                self.optimizer.zero_grad()

                ### Data augmentation
                wavelet1, wavelet2, p1, p2 = generate_mask_wavelet_args(
                    wavelets, dec_lvs, max_dec_lv, seq_len, mask_prob
                )
                x1 = mask_wavelet(x_batch, wavelet1, p1)
                x2 = mask_wavelet(x_batch, wavelet2, p2)
                x2, x2_start = crop(x2, l_min, l_max)

                ### Encoding step
                # encoder output - (N, L, C)
                x1 = select_sub_series(self.encoder(x1), x2.size(1), x2_start)
                x2 = self.encoder(x2)
                # projhead + maxpool output - (N, C)
                x1 = self.proj_head(self.maxpool(x1))
                x2 = self.proj_head(self.maxpool(x2))

                loss = self.loss_fn(x1, x2)
                loss.backward()
                self.optimizer.step()

                lr_scheduler.step()


def eval_classification(
    model, X_train, y_train, X_test, y_test,
    batch_size=256, max_samples=100000, max_iter=1000000, seed=0,
):
    if X_train.shape[0] > max_samples:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_samples, random_state=seed, stratify=y_train,
        )
    X_train_repr = model.encode(X_train, batch_size)
    X_test_repr = model.encode(X_test, batch_size)

    lr = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(random_state=seed, max_iter=max_iter),
        ),
    )
    lr.fit(X_train_repr, y_train)
    y_pred = lr.predict(X_test_repr)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return acc, macro_f1
