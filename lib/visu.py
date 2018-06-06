from const import *
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt


def traverse_latent_space(vae, origin_frame, final_frame, total_ite):
    with torch.no_grad():
        origin_frame = origin_frame.view(-1, 3, WIDTH, HEIGHT)
        final_frame = final_frame.view(-1, 3, WIDTH, HEIGHT) 
        res = final_frame
        origin_z = vae(origin_frame, encode=True)
        final_z = vae(final_frame, encode=True)
        for i in range(0, 50):
            i /= 50
            translat_img = (i * origin_z) + (1 - i) * final_z
            res = torch.cat((res, vae.decode(translat_img)))
            
    res = torch.cat((res, origin_frame))
    save_image(res, 'results/vae/sample_traverse_{}.png'.format(total_ite))

def create_traverse_latent(vae, version):
    imgs = []
    for i in range(LATENT_VEC):
        sample = torch.full((LATENT_VEC,), 0).to(DEVICE)
        for j in range(20):
            sample[i] = sample[i] + 0.05
            final_sample = vae.decode(sample).cpu()
            imgs.append(final_sample[0])
    save_image(imgs,
        'results/vae/sample_{}.png'.format(version))
    

def create_img(vae, version):
    with torch.no_grad():
        sample = torch.randn(64, LATENT_VEC).to(DEVICE)
        final_sample = vae.decode(sample).cpu()
    save_image(final_sample,
        'results/vae/sample_{}.png'.format(version))


def create_img_recons(vae, original_frames, version):
    with torch.no_grad():
        final_sample, _, _ = vae(original_frames)
    save_image(torch.cat((original_frames, final_sample)),
        'results/vae/sample_{}.png'.format(version))
