import pywt
import numpy as np
import torch
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
from augmentations import (
    generate_mask_wavelet_args,
    mask_wavelet,
    rand_crop,
    select_sub_series,
)


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
            list(self.encoder.parameters()) + list(self.proj_head.parameters())
        )
        self.lr = lr
        self.lambda_coeff = lambda_coeff
        self.device = device

    def maxpool(self, x):
        """ Input:  (N, C_in, L)
            Output: (N, C_out)
        """
        return F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

    def encode(self, data_npy, batch_size=16):
        """ Input:  (N, C_in, L)
            Output: (N, C_out)
        """
        data_loader = DataLoader(
            UnlabeledDataset(data_npy),
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
        l_min=0.5,
        l_max=1.0,
        mask_prob=0.1,
        max_dec_lv=4,
        wavelets=["haar", "db2"],
    ):
        self.loss_fn = BarlowTwinsLoss(
            batch_size, lambda_coeff=self.lambda_coeff,
        )

        train_loader = DataLoader(
            UnlabeledDataset(train_data_npy),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        n_samples, _, seq_len = train_data_npy.shape
        n_iters_per_epoch = int(np.ceil(n_samples / batch_size))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=n_epochs,
            steps_per_epoch=n_iters_per_epoch,
        )

        # params for `generate_mask_wavelet_args`
        dec_lvs = {
            wavelets[0]: pywt.dwt_max_level(seq_len, wavelets[0]),
            wavelets[1]: pywt.dwt_max_level(seq_len, wavelets[1]),
        }

        for epoch in range(n_epochs):

            for x_batch in train_loader:
                ### Data augmentation
                wavelet1, wavelet2, p1, p2 = generate_mask_wavelet_args(
                    wavelets, dec_lvs, max_dec_lv, mask_prob
                )
                x1 = mask_wavelet(x_batch, wavelet1, p1)
                x2 = mask_wavelet(x_batch, wavelet2, p2)
                x2, x2_start = rand_crop(x2, l_min, l_max)

                ### Encoding step
                # encoder output - (N, C, L)
                x1 = select_sub_series(
                    self.encoder(x1.to(self.device)), x2.size(2), x2_start,
                )
                x2 = self.encoder(x2.to(self.device))
                # projhead + maxpool output - (N, C)
                x1 = self.proj_head(self.maxpool(x1))
                x2 = self.proj_head(self.maxpool(x2))

                self.optimizer.zero_grad()
                loss = self.loss_fn(x1, x2)
                loss.backward()
                self.optimizer.step()

                lr_scheduler.step()


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        """ X: (n_samples, channels, length)
        """
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def eval_classification(
    model, X_train, y_train, X_test, y_test,
    batch_size=16, max_samples=100000, max_iter=1000000, seed=0,
):
    if X_train.shape[0] > max_samples:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_samples, random_state=seed, stratify=y_train,
        )
    X_train_repr = model.encode(X_train, batch_size)
    X_test_repr = model.encode(X_test, batch_size)

    lr = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(random_state=seed, max_iter=max_iter)),
    )

    lr.fit(X_train_repr, y_train)
    y_pred = lr.predict(X_test_repr)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return acc, macro_f1
