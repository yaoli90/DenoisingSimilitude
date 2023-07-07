import matplotlib.pyplot as plt
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from torchvision.utils import save_image
import os
import numpy as np

            
def plot_radar(psnr, f):
    plt.plot(\
        np.sin(np.linspace(0,2*np.pi,1000)), \
        np.cos(np.linspace(0,2*np.pi,1000)), 'k')
    for j in range(4):
        plt.plot(\
            np.sin(np.linspace(0,2*np.pi,1000))*(j+1)*.2, \
            np.cos(np.linspace(0,2*np.pi,1000))*(j+1)*.2, \
            '--', color=[.1,.1,.1,.5], linewidth=.5)
    for j in range(12):
        plt.plot([0,np.sin(j*np.pi/6)],[0,np.cos(j*np.pi/6)],\
            '--', color=[.1,.1,.1,.5], linewidth=.5)
    angle = np.linspace(0, 2*np.pi, len(psnr))
    plt.fill(np.sin(angle)*psnr, np.cos(angle)*psnr,'g',alpha=.3,edgecolor='green')
    plt.axis('off')
    plt.axis('equal')
            
            
    _, f_name = os.path.split(f)

    plt.savefig(os.path.join('./result', "radar_"+f_name))
    plt.close("all")
    


def plot_result(ISource, INoisy, Out, Denoised, Adv_Noisy, Adv_Out, Adv_Denoised, f):
    psnr = peak_signal_noise_ratio(ISource, Denoised)
    Adv_psnr = peak_signal_noise_ratio(ISource, Adv_Denoised)
    psnr_noisy = peak_signal_noise_ratio(ISource, INoisy)
    Adv_noisy_psnr = peak_signal_noise_ratio(ISource, Adv_Noisy)

    plt.figure()

    plt.subplot(2,4,1)
    plt.imshow(INoisy, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.title('Gauss+img')
    
    plt.subplot(2,4,3)
    plt.imshow(INoisy - ISource, cmap='gray')
    plt.axis('off')
    plt.title('Gauss noise')
    
    plt.subplot(2,4,4)
    plt.imshow(Denoised, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.title('Denoised\n %f'%psnr)
    
    plt.subplot(2,4,5)
    plt.imshow(Adv_Noisy, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.title('Adv+gauss+img')
    
    plt.subplot(2,4,6)
    plt.imshow(Adv_Noisy - INoisy, cmap='gray')
    plt.axis('off')
    plt.title('Adv noise')
    
    plt.subplot(2,4,7)
    plt.imshow(Adv_Noisy - ISource, cmap='gray')
    plt.axis('off')
    plt.title('Adv noise+Gauss')
   
    plt.subplot(2,4,8)
    plt.imshow(Adv_Denoised, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.title('Adv denoised\n %f'%Adv_psnr)

    plt.tight_layout()
    _, f_name = os.path.split(f)

    plt.savefig(os.path.join('./result', f_name))
    plt.close("all")
    plt.imsave(os.path.join('./result', "denoised_"+f_name), Denoised, cmap='gray', vmax=1, vmin=0)
    plt.imsave(os.path.join('./result', "adv_denoised_"+f_name), Adv_Denoised, cmap='gray', vmax=1, vmin=0)
    
    


