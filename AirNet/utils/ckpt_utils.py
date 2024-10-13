import numpy as np
import torch
import os
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
import subprocess

from net.Embedder import Embedder

def load_embedder_ckpt(device, freeze_model=True,
                                  combine_type = ['clear', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']):
    ckpt_name = '../OneRestore/ckpts/embedder_model.tar'

    if torch.cuda.is_available():
        model_info = torch.load(ckpt_name)
    else:
        model_info = torch.load(ckpt_name, map_location=torch.device('cpu'))

    print('==> loading existing Embedder model:', ckpt_name)
    model = Embedder(combine_type)
    model.load_state_dict(model_info)
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



def load_latest_ckpt(net, ckpt_path):
    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith('epoch_') and f.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint found. Starting from scratch.")
        return net, 0

    epoch_values = [int(f.split('_')[1].split('.')[0]) for f in checkpoint_files]

    latest_epoch = max(epoch_values)
    latest_checkpoint = f'epoch_{latest_epoch}.pth'

    print(f"Loading checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    print(checkpoint.keys())
    import sys
    sys.exit(0)

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    net.load_state_dict(checkpoint['model_state_dict'])

    return net, latest_epoch 