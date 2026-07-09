import random
import numpy as np
import torch
from pathlib import Path

### local sripts
from learner import TSReprLearner, eval_classification
from encoder import CausalConvEncoder, MLPHead
from utils import load_benchmark_data, save_model, load_model


DEVICE = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    seed = 0
    set_random_seed(seed)

    data_path = "./datasets/USC_HAD.npy"
    X_train, y_train, X_test, y_test = load_benchmark_data(data_path)
    _, channels, seq_len = X_train.shape

    if seq_len > 500: max_dec_lv = 8
    elif seq_len > 100: max_dec_lv = 6
    else: max_dec_lv = 4

    model_args = {
        "encoder": {
            "in_dim": channels,
            "out_dim": 320,
            "hidden_dim": 80,
            "kernel_size": 3,
            "n_blocks": 6,
            "activation": "elu",
            "dropout": 0.1,
        },
        "proj_head": {
            "in_dim": 320,
            "hidden_dim": [1024, 1024],
            "out_dim": 1024,
            "bias": False,
            "b_norm": True,
        }
    }

    encoder = CausalConvEncoder(**model_args["encoder"])
    proj_head = MLPHead(**model_args["proj_head"])

    model = TSReprLearner(
        encoder,
        proj_head,
        lr=0.003,
        lambda_coeff=1e-3,
        device=DEVICE,
    )

    model.pretrain(
        X_train,
        n_epochs=50,
        batch_size=256,
        l_min=0.5,
        l_max=1.0,
        mask_prob=0.1,
        max_dec_lv=max_dec_lv,
        wavelets=["haar", "db2"],
    )

    ### save model into `pretrained_dir` folder
    name = data_path.split("/")[-1].split(".")[0] + "_" + str(seed)
    pretrained_dir = Path("./saved_models")/name
    save_model(model, model_args, pretrained_dir)

    acc, macro_f1 = eval_classification(model, X_train, y_train, X_test, y_test)
    print(f"acc: {acc:.4f} | macro-f1: {macro_f1:.4f}")

    ### load model from `pretrained_dir`
    # _encoder, _proj_head = load_model(pretrained_dir, CausalConvEncoder, MLPHead, DEVICE)
    # pretrained_model = TSReprLearner(_encoder, _proj_head)

    # acc, macro_f1 = eval_classification(model, X_train, y_train, X_test, y_test)
    # print(f"acc: {acc:.4f} | macro-f1: {macro_f1:.4f}")
