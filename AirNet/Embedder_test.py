import os, time, argparse
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from AirNet.utils.ckpt_utils import load_embedder_ckpt

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = load_embedder_ckpt(device, freeze_model=True)
    with torch.no_grad():
        print(args)
        result = embedder(args,'text_encoder')
        print(result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Running")

    parser.add_argument('arguments', nargs='+')

    args = parser.parse_args()

    main(args.arguments)
