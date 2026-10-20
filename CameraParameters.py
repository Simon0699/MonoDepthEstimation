import torch 
from PIL import Image
import torchvision.transforms.functional as TF
#rodriguez formula
def rotation_axis_angle(k, theta): # k = (B, 3) theta = (B) radians
    B = k.shape[0]
    kx = k[:, 0] #B
    ky = k[:, 1] #B
    kz = k[:, 2] #B
    K = torch.stack([torch.zeros(B), -kz, ky, kz, torch.zeros(B), -kx, -ky, kx, torch.zeros(B)]).T.reshape(B, 3,3)
    R = torch.eye(3).repeat(B,1,1) + torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * K + (1-torch.cos(theta)).unsqueeze(-1).unsqueeze(-1) * K @ K
    return R # B, 3, 3

def get_camera_params(images, corner_annotations): # list of strings, (B, 25, 2)
    image_tensors = []
    for i in images: 
        t = TF.pil_to_tensor(Image.open(i))
        image_tensors.append(TF.resize(t, (224,224)))
    images = torch.stack(image_tensors)
    # (B, 3, 224, 224)
    B = images.shape[0]
    init_rotation_axes = torch.nn.Parameter(torch.randn((B, 3)))
    init_angles = torch.nn.Parameter(0.1 * torch.ones(B))
    init_translations = torch.nn.Parameter(torch.tensor([-2,-2,10.0]).unsqueeze(0).repeat(B, 1))

    camera_parameters = torch.nn.Parameter(torch.tensor([224.0, 224, 112, 112]))
    x, y = torch.meshgrid(torch.arange(5), torch.arange(5), indexing = "xy")
    z = torch.zeros(5,5)
    points_irl = torch.stack((x,y,z)).flatten(1).T # (25,3)
    iterations = 5000

    camera_param_dict = {"params": camera_parameters, "lr" : 1}
    non_camera_param_dict = {"params": [init_rotation_axes, init_angles, init_translations], "lr": 0.01}
    loss_stats = []
    optim = torch.optim.Adam([camera_param_dict, non_camera_param_dict])
    for i in range(iterations):
        optim.zero_grad()
        fx, fy, cx, cy = camera_parameters
        K = torch.stack([fx, torch.tensor(0.0), cx, torch.tensor(0), fy, cy, torch.tensor(0), torch.tensor(0), torch.tensor(1)]).reshape(3,3)
        norm_rotation_axes = init_rotation_axes/(init_rotation_axes.norm(dim = 1, keepdim = True))
        rotation_matrices = rotation_axis_angle(norm_rotation_axes, init_angles)
        tpr3 = (rotation_matrices @ points_irl.T).transpose(1,2) # B, 25, 3
        tpr3 = tpr3 + init_translations.unsqueeze(1)# B, 25, 3 + B, 1, 3 = B, 25, 3
        homr3 = K @ tpr3.transpose(1, 2) # B, 3, 25
        homr3 = homr3.transpose(0,1) # 3, B, 25 
        proj_points = torch.stack((homr3[0]/homr3[2], homr3[1]/homr3[2]), -1) # B, 25, 2
        loss = ((corner_annotations - proj_points)**2).mean()
        loss_stats.append(loss.item())
        loss.backward()
        optim.step()
    return K.detach(), loss_stats
