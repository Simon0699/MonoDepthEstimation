import torch 
import torch.nn as nn
class Losses(nn.Module):
    ssim_patchify = nn.Sequential(nn.ReflectionPad2d(1), nn.AvgPool2d((3,3), 1))
    def __init__(self):
        super().__init__()
    def SSIM(self, images1, images2): # B, 3, 224, 224
        x_patches = Losses.ssim_patchify(images1) # B, 3, 224, 224
        y_patches = Losses.ssim_patchify(images2) # B, 3, 224, 224
        c1 = (0.01)**2
        c2 = (0.03)**2
        x_var = Losses.ssim_patchify(images1**2) - x_patches**2
        y_var = Losses.ssim_patchify(images2**2) - y_patches**2
        cov = Losses.ssim_patchify(images1 * images2) - x_patches * y_patches
        num = (2 * x_patches * y_patches + c1) * (2 * cov + c2)
        denom = (x_patches**2 + y_patches**2 + c1) * (x_var + y_var + c2)
        return num/denom #B, 3, 224, 224







    def smooth_depth_loss(self, images, depth_maps): # (B, 3, 224, 224), (B, 224, 224)
        B = images.shape[0]

        inverse = 1/depth_maps
        inverse = inverse/(inverse.mean(dim = (1,2)).view(B, -1, -1))
        xdiff = (inverse[:, 2:, :] - inverse[:, :-2, :])/2
        ydiff = (inverse[:, :, 2:] - inverse[:, :, :-2])/2
        xdiff_mag_depth = xdiff.mean(dim = (1,2)) # (B) collapse over x and y 
        ydiff_mag_depth = ydiff.mean(dim = (1,2)) # (B)

        image_x_diff = (images[:, :, 2:, : ] - images[:, :, :-2, :])/2
        image_y_diff = (images[:, :, :, 2:] - images[:, :, :, :-2])/2

        xdiff_mag_image = image_x_diff.mean(dim = (1,2,3))
        ydiff_mag_image = image_y_diff.mean(dim = (1,2,3))

        loss = xdiff_mag_depth * torch.exp(-xdiff_mag_image) + ydiff_mag_depth * torch.exp(-ydiff_mag_image)
        return loss
