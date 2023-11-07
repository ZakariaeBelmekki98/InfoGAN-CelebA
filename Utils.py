import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_images(images, num_images, nrow):
    images = (images + 1)/2
    images = image.detach().cpu()
    grid = make_grid(images[:num_images], nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


