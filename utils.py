import json
import numpy as np
import random
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_benchmark_data(path):
    # shape: (n_sample, channels, length)
    data = np.load(path, allow_pickle=True).item()
    train_data, train_label = data["train_data"], data["train_label"]
    test_data, test_label = data["test_data"], data["test_label"]

    # check if data contains validation set. if true, merge training and validation set
    name = path.split("/")[-1].split(".")[0].lower()
    if "val_data" in data and name != "sleep":
        val_data, val_label = data["val_data"], data["val_label"]
        train_data = np.concatenate([train_data, val_data])
        train_label = np.concatenate([train_label, val_label])

    # label encoder (only for "USC_HAD")
    if name == "usc_had":
        label_encoder = LabelEncoder()
        train_label = label_encoder.fit_transform(train_label)
        test_label = label_encoder.transform(test_label)

    return train_data, train_label, test_data, test_label

def get_labeled_data(X, y, n=1, seed=0):
    """
    sample a small subset of data for linear probing
    (the return data dist is uniform)
    """
    random.seed(seed)
    np.random.seed(seed)

    _X, _y = [], []
    for c in np.unique(y):
        _X.append(np.random.permutation(X[y == c])[:n])
        _y.append(y[y == c][:n])

    return np.concatenate(_X), np.concatenate(_y)

def get_labeled_data_incemental(samples, labels, percentage=[0.1, 0.05, 0.01], seed=0):
    """
    Incrementally sample small subsets of the data.
    `percentage=[0.1, 0.05, 0.01]` -> sample 10%, 5% and 1% of the data
     and store them in `labeled_ds`.
     (the distribution is roughly the same as the original data)

    Use `labeled_ds["1%"] (or labeled_ds["5%"]) to access the data.
    """
    random.seed(seed)
    np.random.seed(seed)

    labeled_ds = {}
    X, y = samples, labels
    prev_p = 1.0

    for p in percentage:
        _X, _, _y, _ = train_test_split(
            X, y, train_size=p/prev_p, stratify=y, random_state=seed,
        )
        labeled_ds[f"{int(p*100)}%"] = {"X": _X, "y": _y}
        X, y, prev_p = _X, _y, p

    return labeled_ds


def save_model(model, args, save_path):
    save_path = Path(save_path)
    # create dir for storing saved models
    Path.mkdir(save_path, exist_ok=True, parents=True)
    # save trained encoder and output projector
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
