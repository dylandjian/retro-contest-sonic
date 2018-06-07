from const import *
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt


def traverse_latent_space(vae, origin_frame, final_frame, total_ite):
    " Linear latent space interpolation between 2 frames "

    with torch.no_grad():
        origin_frame = origin_frame.view(-1, 3, WIDTH, HEIGHT)
        final_frame = final_frame.view(-1, 3, WIDTH, HEIGHT) 
        res = final_frame
        origin_z = vae(origin_frame, encode=True)
        final_z = vae(final_frame, encode=True)
        number_frames = 50

        for i in range(0, number_frames):
            i /= number_frames
            translat_img = (i * origin_z) + (1 - i) * final_z
            res = torch.cat((res, vae.decode(translat_img)))
            
    res = torch.cat((res, origin_frame))
    save_image(res, 'results/vae/sample_traverse_{}.png'.format(total_ite))

    
def create_img_recons(vae, original_frames, version):
    """ Save the image and its reconstruction """

    with torch.no_grad():
        final_sample, _, _ = vae(original_frames)
    save_image(torch.cat((original_frames, final_sample)),
        'results/vae/sample_{}.png'.format(version))
