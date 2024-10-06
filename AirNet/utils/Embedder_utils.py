import numpy as np
import torch
import os
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd

from net.Embedder import Embedder

def load_embedder_ckpt(device, freeze_model=True,
                                  combine_type = ['clear', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']):
    ckpt_name = '../../OneRestore/ckpts/embedder_model.tar'

    print('==> Initialize Embedder model.')
    model = Embedder(combine_type)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if freeze_model:
        freeze(model)

    return model

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None