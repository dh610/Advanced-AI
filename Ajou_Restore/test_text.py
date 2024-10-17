import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils_CDD import DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from text_net.model import AirNet

import os
import re
import csv
import subprocess

def extract_info_from_filename(filename):
    # epoch_%d_l1_%.4f_cl_%.2f.pth에서 epoch, l1_loss, contrast_loss 추출
    match = re.match(r'epoch_(\d+)_l1_(\d+\.\d+)_cl_(\d+\.\d+)\.pth', filename)
    if match:
        epoch = int(match.group(1))
        l1_loss = float(match.group(2))
        contrast_loss = float(match.group(3))
        return epoch, l1_loss, contrast_loss
    return None, None, None

def save_to_csv(csv_path, data):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'l1_loss', 'contrast_loss', 'psnr', 'ssim'])
        for row in data:
            writer.writerow(row)

def test_and_save_results(net, dataset, ckpt_files, csv_path, task="derain"):
    results = []
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    for ckpt_file in ckpt_files:
        # 파일에서 epoch, l1_loss, contrast_loss 정보 추출
        epoch, l1_loss, contrast_loss = extract_info_from_filename(ckpt_file)
        if epoch is None:
            continue

        # 모델 로드
        net.load_state_dict(torch.load(ckpt_file, map_location=torch.device(opt.cuda)))
        net.eval()

        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([degraded_name], degradation, degrad_patch, clean_patch, text_prompt) in tqdm(testloader):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                restored = net(x_query=degrad_patch, x_key=degrad_patch, text_prompt=text_prompt)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

        # 성능 결과 저장
        print("l1_loss: %.4f, contrast_loss: %.2f, PSNR: %.2f, SSIM: %.4f" % (l1_loss, contrast_loss, psnr.avg, ssim.avg))
        results.append([epoch, l1_loss, contrast_loss, psnr.avg, ssim.avg])
    
    # CSV 저장
    save_to_csv(csv_path, results)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1, help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')
    parser.add_argument('--derain_path', type=str, default="data/CDD-11_test_100/", help='save path of test raining images')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    opt = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)
    opt.batch_size = 7

    # Dataset and Model
    derain_set = DerainDehazeDataset(opt)
    net = AirNet(opt).cuda()

    # Checkpoint 파일 목록 가져오기
    ckpt_files = [os.path.join(opt.ckpt_path, f) for f in os.listdir(opt.ckpt_path) if f.endswith('.pth')]
    csv_path = 'result.csv'

    # 성능 측정 및 CSV 저장
    test_and_save_results(net, derain_set, ckpt_files, csv_path, task="derain")

#########################################################################

'''
def test_Denoise(net, dataset, sigma=15):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    print(len(testloader))
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degradation, degrad_patch, clean_patch, text_prompt) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            print(f"Clean patch size: {clean_patch.size()}, Restored size: {degrad_patch.size()}")
            restored = net(x_query=degrad_patch, x_key=degrad_patch, text_prompt = text_prompt)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degradation[0] + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/CDD-11_test_100/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'Denoise.pth'
    elif opt.mode == 1:
        opt.batch_size = 7
        ckpt_path = opt.ckpt_path + 'epoch_510.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'All.pth'

    # denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)

    # Make network
    net = AirNet(opt).cuda()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))
    net.eval()

    if opt.mode == 0:
        print('Start testing Sigma=15...')
        # test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        # test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        # test_Denoise(net, denoise_set, sigma=50)
    elif opt.mode == 1:
        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="derain")
    elif opt.mode == 2:
        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif opt.mode == 3:
        print('Start testing Sigma=15...')
        # test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        # test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        # test_Denoise(net, denoise_set, sigma=50)

        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")
'''