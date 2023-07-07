import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from used_attacks import denoising_PGD
from torchvision import transforms
from utils import *
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from visualization import plot_result, plot_radar

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/DnCNN-S-25", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

'''
    Output: normalized psnr for the radar diagram
'''
def radar(model, ISource, INoisy, Adv_Noisy, N_rotate=100):
    u1_norm = np.linalg.norm((INoisy - ISource).cpu().numpy())
    u2_norm = np.linalg.norm((Adv_Noisy - INoisy).cpu().numpy())
    u1 = (INoisy - ISource).cpu().numpy().flatten() / u1_norm
    u2 = (Adv_Noisy - INoisy).cpu().numpy().flatten() / u2_norm
    un = 1/np.sqrt(np.dot(u1,u2)**2 + 1)*u2 - np.dot(u1,u2)/np.sqrt(np.dot(u1,u2)**2 + 1)*u1

    psnr = np.zeros(N_rotate)
    for i in range(N_rotate):
        theta = np.pi*2*i/N_rotate
        Adv_rotate = torch.Tensor((np.sin(theta)*u1 + np.cos(theta)*un)*u2_norm).reshape(Adv_Noisy.shape).cuda()+INoisy
        
        Adv_Out_rotate = model(Adv_rotate)
        Adv_Denoised_rotate = torch.clamp(Adv_rotate-Adv_Out_rotate, 0., 1.)
        psnr[i] = peak_signal_noise_ratio(ISource.cpu().numpy(), Adv_Denoised_rotate.cpu().detach().numpy())
    
    Denoised = INoisy - model(INoisy)
    psnr_max = peak_signal_noise_ratio(ISource.cpu().numpy(), Denoised.cpu().detach().numpy())
    psnr_diff = np.clip(psnr_max-psnr, 0, None)
    return (psnr_diff-np.min(psnr_diff))/(np.max(psnr_diff)-np.min(psnr_diff))
    

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    # process data
    for f in files_source:
        # load test image
        Img = cv2.imread(f)
        Img = np.float32(Img[:,:,0])/255.
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        
        ISource = torch.Tensor(Img) # clean image
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.) # Gaussian noise
        INoisy = ISource + noise # input image with Gaussian noise
        
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): 
            Out = model(INoisy) # estimated noise residule
            Denoised = torch.clamp(INoisy-Out, 0., 1.) # denoised Gaussian noise input image
        
        # adversarial attack
        denoising_attack = denoising_PGD(model, steps=5, eps=3/255, alpha=2/255)
        Adv_Noisy = denoising_attack.attack(INoisy) # adversarial sample for input image
        Adv_Out = model(Adv_Noisy) # estimated noise residule of adversarial sample
        Adv_Denoised = torch.clamp(Adv_Noisy-Adv_Out, 0., 1.) # denoised adversarial input image
        
        
        print(f, \
            "denoised PSNR:",  \
            peak_signal_noise_ratio(ISource.cpu().numpy(), Denoised.cpu().numpy()), \
            "denoised-adv PSNR:", \
            peak_signal_noise_ratio(ISource.cpu().numpy(), Adv_Denoised.cpu().detach().numpy()))
            
            
        plot_result(\
            np.squeeze(ISource.cpu().numpy()),
            np.squeeze(INoisy.cpu().numpy()),
            np.squeeze(Out.cpu().numpy()),
            np.squeeze(Denoised.cpu().numpy()),
            np.squeeze(Adv_Noisy.cpu().numpy()),
            np.squeeze(Adv_Out.cpu().detach().numpy()),
            np.squeeze(Adv_Denoised.cpu().detach().numpy()),
            f)
            
            
        rotate_psnr = radar(model, ISource, INoisy, Adv_Noisy)
        plot_radar(rotate_psnr, f)

    
if __name__ == "__main__":
    main()
