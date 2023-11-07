import torch
from torch import nn

def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
        torch.nn.init.constant_(layer.bias, 0)

class Generator(nn.Module):
    def __init__(self, z_dim=64, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_block(z_dim, hidden_dim * 8),
            self.make_block(hidden_dim * 8, hidden_dim * 4, stride=1),
            self.make_block(hidden_dim * 4, hidden_dim * 2, kernel_size=3),
            self.make_block(hidden_dim * 2, hidden_dim, kernel_size=3),
            self.make_block(hidden_dim, im_chan, final_layer=True)
        )

        
    def make_block(self, input_dims, output_dims, kernel_size=4, stride=2, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                nn.BatchNorm2d(output_dims),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, im_chan=3, c_dim=5, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.c_dim = c_dim
        self.body = nn.Sequential(
            self.make_block(im_chan, hidden_dim),
            self.make_block(hidden_dim, hidden_dim * 2),
            self.make_block(hidden_dim * 2, hidden_dim * 4),
        )
        self.disc = self.make_block(hidden_dim*4, 1, kernel_size=5, final_layer=True)
        self.q = nn.Sequential(
            self.make_block(hidden_dim * 4, hidden_dim * 4),
            self.make_block(hidden_dim * 4, 2*c_dim, kernel_size=1, final_layer=True)
        )
        
    def make_block(self, input_dim, output_dim, kernel_size=4, stride=2, final_layer=False):
        if final_layer:
            return nn.Conv2d(input_dim, output_dim, kernel_size, stride)
        else:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
    def forward(self, image):
        intermediate_pred = self.body(image)
        disc_pred = self.disc(intermediate_pred)
        q_pred = self.q(intermediate_pred)
        return disc_pred.view(len(disc_pred), -1), q_pred.view(len(q_pred), -1)
