import subprocess
from tqdm import tqdm

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils_CDD import TrainDataset
from net.model import AirNet

from option import options as opt

from utils.Embedder_utils import load_embedder_ckpt

de_arr = [
    'haze', 'rain', 'low',      \
    'haze_rain', 'low_haze',    \
    'low_rain', 'low_haze_rain' \
]

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    embedder = load_embedder_ckpt(opt.cuda, freeze_model=True)

    # Network Construction
    net = AirNet(opt, embedder.out_dim).cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            optimizer.zero_grad()

            embedding_vectors = []

            cnt = 0
            for id in de_id:

                embedding_vector = torch.zeros(embedder.out_dim)
                
                id = id.unsqueeze(0)

                if cnt == 0:
                    embedding_vector = embedder(id, 'text_idx_encoder')
                print(embedding_vector.shape())

                embedding_vectors.append(embedding_vector)

                cnt = (cnt + 1) % 4

            embedding_vectors = torch.stack(embedding_vectors)

            if epoch < opt.epochs_encoder:
                _, output, target, _ = net.E(x_query=degrad_patch_1, x_key=degrad_patch_2, text_embedding=embedding_vectors)
                contrast_loss = CE(output, target)
                loss = contrast_loss
            else:
                restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2, text_embedding=text_embedding)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, clean_patch_1)
                loss = l1_loss + 0.1 * contrast_loss

            # backward
            loss.backward()
            optimizer.step()

        if epoch < opt.epochs_encoder:
            print(
                'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                    epoch, contrast_loss.item(),
                ), '\r', end='')
        else:
            print(
                'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n' % (
                    epoch, l1_loss.item(), contrast_loss.item(),
                ), '\r', end='')

        GPUS = 1
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if GPUS == 1:
                torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            else:
                torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')

        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 60))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
