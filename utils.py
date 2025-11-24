import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """ data shape: (n_samples, length, features)
        """
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def load_benchmark_data(data_path):
    data_name = data_path.split("/")[-1].split(".")[0]
    # data shape: (n_samples, features, length)
    data = np.load(data_path, allow_pickle=True).item()
    # transpose to (n_samples, length, features)
    train_data, train_label = data["train_data"], data["train_label"]
    test_data, test_label = data["test_data"], data["test_label"]

    # check if the data contain validation set
    #  if true, merge training and validation sets
    if "val_data" in data and data_name != "Sleep":
        val_data, val_label = data["val_data"], data["val_label"]
        train_data = np.concatenate([train_data, val_data], axis=0)
        train_label = np.concatenate([train_label, val_label], axis=0)

    # label encoder (only for "USC_HAD")
    if data_name == "USC_HAD":
        le = LabelEncoder()
        train_label = le.fit_transform(train_label)
        test_label = le.transform(test_label)

    # transpose to (n samples, length, features)
    return (
        train_data.transpose(0, 2, 1),
        train_label,
        test_data.transpose(0, 2, 1),
        test_label,
    )

def get_labeled_data(X, y, n=1, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    X_s, y_s = [], []
    for c in np.unique(y):
        X_s.append(np.random.permutation(X[y == c])[:n])
        y_s.append(y[y == c][:n])
    return np.concatenate(X_s, axis=0), np.concatenate(y_s, axis=0)

def get_labeled_data_ratio(
    X_train_all,
    y_train_all,
    ratios=[0.2, 0.15, 0.10, 0.05, 0.01],
    seed=0,
):
    random.seed(seed)
    np.random.seed(seed)

    labeled_ds = {}
    train_data, train_label = X_train_all, y_train_all
    prev_ratio = 1.0

    for ratio in ratios:
        x, _, y, _ = train_test_split(
            train_data, train_label,
            train_size=ratio/prev_ratio,
            stratify=train_label,
            random_state=seed,
        )
        labeled_ds[int(ratio*100)] = {"X": x, "y": y}
        train_data, train_label = x, y
        prev_ratio = ratio

    return labeled_ds

def save_model(model, args, save_path):
    save_path = Path(save_path)
    # create dir for storing saved models
    Path.mkdir(save_path, exist_ok=True, parents=True)
    # save trained encoder and output projector
    if isinstance(model.encoder, torch.optim.swa_utils.AveragedModel):
        torch.save(model.encoder.module.state_dict(), f=save_path/"encoder")
    else:
        torch.save(model.encoder.state_dict(), f=save_path/"encoder")
    torch.save(model.proj_head.state_dict(), f=save_path/"proj_head")
    # save hyperparams
    with open(save_path/"encoder_args.json", "w") as f:
        json.dump(args["encoder"], f)
    with open(save_path/"proj_head_args.json", "w") as f:
        json.dump(args["proj_head"], f)

def load_model(save_path, encoder_cls, proj_head_cls, device=torch.device("cpu")):
    save_path = Path(save_path)
    # load hyperparams
    with open(save_path/"encoder_args.json", "r") as f:
        encoder_args = json.load(f)
    # load encoder weights
    encoder = encoder_cls(**encoder_args).to(device)
    encoder.load_state_dict(
        torch.load(save_path/"encoder", weights_only=True, map_location=device)
    )
    with open(save_path/"proj_head_args.json", "r") as f:
        proj_head_args = json.load(f)
    proj_head = proj_head_cls(**proj_head_args).to(device)
    proj_head.load_state_dict(
        torch.load(save_path/"proj_head", weights_only=True, map_location=device)
    )
    return encoder, proj_head