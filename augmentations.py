import random
import numpy as np
import ptwt
import torch


def select_sub_series(x, cropped_len, start_indices):
    indices = start_indices[:, None] + torch.arange(cropped_len)
    return x[torch.arange(x.size(0))[:, None], :, indices].transpose(1, 2)

def rand_crop(x, l_min, l_max):
    batch_size, _, seq_len = x.shape
    cropped_len = int(seq_len * np.random.uniform(low=l_min, high=l_max))
    start_indices = torch.randint(0, seq_len-cropped_len+1, (batch_size,))
    return (
        select_sub_series(x, cropped_len, start_indices),
        start_indices
    )

def mask_wavelet(x, wavelet, mask_probs, mode="reflect"):
    """ input:  (N, C, L)
        output: (N, C, L)
    """
    level = len(mask_probs)
    # coeffs: [cA_n, cD_n, cD_n-1, ..., cD_1]
    # freq: [lowest (cA_n), low (cD_n), ..., mid (cD_n-k), ..., high (cD_1)]
    coeffs = ptwt.wavedec(x, wavelet, level=level, mode=mode)
    # ignore approximation coeff (i == 0)
    masked_coeffs = [coeffs[0]]

    for i in range(level):
        mask = torch.bernoulli(torch.full_like(coeffs[i+1], 1-mask_probs[i]))
        masked_coeffs.append(coeffs[i+1] * mask)

    return ptwt.waverec(masked_coeffs, wavelet)[:, :, :x.size(2)]

def generate_mask_wavelet_args(wavelets, dec_lvs, max_dec_lv, mask_prob):
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
