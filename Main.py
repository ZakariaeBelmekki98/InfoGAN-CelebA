import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from Models import Discriminator, Generator, init_weights

# MACROS 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 64
C_DIM = 10
C_LAMBDA = 0.1
BATCH_SIZE = 128
EPOCHS = 100
LR_G = 0.001
LR_D = 0.0002
CELEBA_DIR = "/home/zak/Downloads/celeba_test/"

if __name__ == "__main__":
    ### Load Data ###
    transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = CelebA(CELEBA_DIR, split='all', download=False, transform=transform)
    dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=16
            )

    ### Initialize models and loss ###
    disc = Discriminator().to(DEVICE)
    gen = Generator(z_dim=Z_DIM+C_DIM).to(DEVICE)

    disc = disc.apply(init_weights)
    gen = gen.apply(init_weights)

    disc_optim = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(0.5, 0.999))
    gen_optim = torch.optim.Adam(gen.parameters(), lr=LR_G, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    info_criterion = lambda c_true, mean, logvar: Normal(mean, logvar.exp()).log_prob(c_true).mean()
    step = 0
    disc_losses = []
    gen_losses = []

    ### Training loop ###
    for epoch in range(EPOCHS):
        for real, _ in tqdm(dataloader):
            batch_size = real.shape[0]
            real = real.to(DEVICE)

            ### Train the discriminator ###
            disc_optim.zero_grad()
            noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
            c_labels = torch.randn(batch_size, C_DIM, device=DEVICE)
            fake = gen(torch.cat([noise, c_labels], dim=1).view(batch_size, -1, 1, 1))
            disc_fake_pred, disc_q_pred = disc(fake.detach())
            disc_q_mean = disc_q_pred[:, :C_DIM]
            disc_q_logvar = disc_q_pred[:, C_DIM:]
            mutual_info = info_criterion(c_labels, disc_q_mean, disc_q_logvar)
            disc_real_pred, _ = disc(real)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2 - C_LAMBDA * mutual_info
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            disc_losses += [disc_loss.item()]

            ### Update the generator ###
            gen_optim.zero_grad()
            disc_fake_pred, disc_q_pred = disc(fake)
            disc_q_mean = disc_q_pred[:, :C_DIM]
            disc_q_logvar = disc_q_pred[:, C_DIM:]
            mutual_info = infor_criterion(c_labels, disc_q_mean, disc_q_logvar)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - C_LAMBDA * mutual_info
            gen_loss.backward()
            gen_optim.step()

            gen_losses += [gen_loss.item()]

            if stetp % 500 == 0 and step > 0:
                gen_mean = sum(gen_losses[-500:]) / 500
                disc_mean = sum(disc_losses[-500:]) / 500
                print("Epoch {}, step {}, Generator loss {} Discriminator loss {}".format(epoch, step, gen_mean, disc_mean))
                        
            step += 1



    
