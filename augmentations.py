import random
import numpy as np
import ptwt
import pywt
import torch


def select_sub_series(x, lenght, index_start):
    indices = index_start[:, None] + np.arange(lenght)
    return x[torch.arange(indices.shape[0])[:, None], indices]

def crop(x, l_min, l_max):
    """ input: (N, L, C)
    """
    batch_size, seq_len, n_features = x.shape
    # get start/end index of the subseries
    cropped_len = int(seq_len * np.random.uniform(low=l_min, high=l_max))
    start_index = np.random.randint(low=0, high=seq_len-cropped_len, size=batch_size)
    # return cropped segment and its start index
    return select_sub_series(x, cropped_len, start_index), start_index

def mask_wavelet(x, wavelet, mask_probs, mode="reflect"):
    """ input:  (N, L, C)
        output: (N, L, C)
    """
    device = x.device
    seq_len = x.size(1)
    level = len(mask_probs)

    wavelet = pywt.Wavelet(wavelet)
    coeffs = ptwt.wavedec(x.transpose(1, 2), wavelet, level=level, mode=mode)
    # ignore approximation coeff (i == 0)
    masked_coeffs = [coeffs[0]]

    for i in range(level):
        p = mask_probs[i]
        mask = np.random.choice(
            [0, 1], size=coeffs[i+1].shape, p=[p, 1-p],
        )
        mask = torch.from_numpy(mask).to(device)
        masked_coeffs.append((coeffs[i+1]*mask).float())

    return ptwt.waverec(
        masked_coeffs,
        wavelet=wavelet,
    )[:, :, :seq_len].transpose(1, 2).float()

def generate_mask_wavelet_args(wavelets, dec_lvs, max_dec_lv, seq_len, mask_prob):
    w1 = wavelets[random.getrandbits(1)]
    w2 = wavelets[random.getrandbits(1)]

    lv = min(dec_lvs[w1], dec_lvs[w2], max_dec_lv)
    mid_lv = lv // 2

    p1 = np.full(lv, fill_value=mask_prob)
    p2 = np.full(lv, fill_value=mask_prob)

    if random.getrandbits(1):
        p1[random.randint(0, mid_lv-1)] = 1.0
        p2[random.randint(mid_lv, lv-1)] = 1.0
    else:
        p1[random.randint(mid_lv, lv-1)] = 1.0
        p2[random.randint(0, mid_lv-1)] = 1.0

    return w1, w2, p1, p2