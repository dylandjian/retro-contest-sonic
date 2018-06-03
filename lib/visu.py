from const import *
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt


def create_shading(vae, version, sample=None):
    if not sample:
        sample = torch.full((LATENT_VEC,), 0).to(DEVICE)
    else:
        img = torch.tensor([sample[0]], dtype=torch.float, device=DEVICE).div(255)
        with torch.no_grad():
            mu, _ = vae.encode(img)
    imgs = []
    for i in range(LATENT_VEC):
        new_sample = mu.clone()
        for j in range(5):
            new_sample[0][i] += (1 - new_sample[0][i]) * 0.2
            with torch.no_grad():
                final_sample = vae.decode(new_sample).cpu().numpy()[0].transpose(1, 2, 0)
            print(final_sample.shape)
            plt.imshow(final_sample, interpolation='nearest')
            plt.show()
            imgs.append(final_sample[0])
        new_sample = mu.clone()
        for j in range(5):
            new_sample[0][i] -= (1 - new_sample[0][i]) * 0.2
            with torch.no_grad():
                final_sample = vae.decode(new_sample).cpu().numpy()[0].transpose(1, 2, 0)
            plt.imshow(final_sample, interpolation='nearest')
            plt.show()
            imgs.append(final_sample[0])
    save_image(imgs,
        'results/vae/sample_{}.png'.format(version))


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
