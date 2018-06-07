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


def sample(seq, pi, mu, sigma):
    """ Sample a latent vector given pi, mu and sigma """

    sampled = torch.sum(pi * torch.normal(mu, sigma), dim=2)
    return sampled.view(seq, LATENT_VEC)


def sample_long_term(vae, lstm, frames, version, total_ite):
    """ Given a frame, tries to predict the next 60 encoded vectors """

    lstm.hidden = lstm.init_hidden(1)

    ## Add first 4 frames, then their reconstruction
    frames_z = vae(frames.view(-1, 3, WIDTH, HEIGHT), encode=True)[0:4]
    result = torch.cat((frames[0:4], vae.decode(frames_z)\
                .view(-1, 3, WIDTH, HEIGHT)))
    z = frames_z[0].view(1, LATENT_VEC)
    with torch.no_grad():
        for i in range(1, 60):
            new_state = torch.cat((z, torch.full((1, 1), 1, device=DEVICE) / ACTION_SPACE_DISCRETE), dim=1)
            pi, sigma, mu = lstm(new_state.view(1, 1, LATENT_VEC + 1))
            z = sample(1, pi, mu, sigma)
            result = torch.cat((result, vae.decode(z).view(-1, 3, WIDTH, HEIGHT)))
    save_image(result, "results/lstm/test-{}-{}.png".format(version, total_ite))
