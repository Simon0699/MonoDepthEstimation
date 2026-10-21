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
def proj_residuals(camera_parameters, init_rotation_axes, init_angles, init_translations, corner_annotations, points_irl):
    fx, fy, cx, cy = camera_parameters
    K = torch.stack([fx, torch.tensor(0.0), cx, torch.tensor(0), fy, cy, torch.tensor(0), torch.tensor(0), torch.tensor(1)]).reshape(3,3)
    norm_rotation_axes = init_rotation_axes/(init_rotation_axes.norm(dim = 1, keepdim = True))
    rotation_matrices = rotation_axis_angle(norm_rotation_axes, init_angles)
    tpr3 = (rotation_matrices @ points_irl.T).transpose(1,2) # B, 25, 3
    tpr3 = tpr3 + init_translations.unsqueeze(1)# B, 25, 3 + B, 1, 3 = B, 25, 3
    homr3 = K @ tpr3.transpose(1, 2) # B, 3, 25
    homr3 = homr3.transpose(0,1) # 3, B, 25 
    proj_points = torch.stack((homr3[0]/homr3[2], homr3[1]/homr3[2]), -1) # B, 25, 2
    loss = ((corner_annotations - proj_points))
    return loss
def get_camera_params(images, corner_annotations): # list of strings, (B, 25, 2)\
    lam = 0.01 #damping factor
    lr =  0.1 # learning rate
    image_tensors = []
    for i in images: 
        t = TF.pil_to_tensor(Image.open(i))
        image_tensors.append(TF.resize(t, (224,224)))
    images = torch.stack(image_tensors)
    # (B, 3, 224, 224)
    B = images.shape[0]
    init_rotation_axes = (torch.randn((B, 3)))
    init_angles = (0.1 * torch.ones(B))
    init_translations = (torch.tensor([-2,-2,10.0]).unsqueeze(0).repeat(B, 1))

    camera_parameters = (torch.tensor([224.0, 224, 112, 112]))
    x, y = torch.meshgrid(torch.arange(5), torch.arange(5), indexing = "xy")
    z = torch.zeros(5,5)
    points_irl = torch.stack((x,y,z)).flatten(1).T # (25,3)
    iterations = 100

    
    loss_stats = []
    JAC_FN = torch.func.jacrev(proj_residuals, (0,1,2,3))
    for i in range(iterations):
        loss = (proj_residuals(camera_parameters, init_rotation_axes, init_angles, init_translations, corner_annotations, points_irl))



        jacobian = JAC_FN(camera_parameters, init_rotation_axes, init_angles, init_translations, corner_annotations, points_irl)
        full_jacobian = torch.cat([i.flatten(3) for i in jacobian], dim = 3).reshape(B*25*2, -1)
        ##LM STEP
        JTJ = full_jacobian.T @ full_jacobian
        left = (JTJ + lam * torch.diag(torch.diag(JTJ)))
        right = -(full_jacobian.T @ loss.flatten())
        update = torch.linalg.solve(left, right)

        camera_parameters += lr * update[0:4]
        init_rotation_axes += lr * update[4:4+3*B].reshape(B,3)
        init_angles += lr * update[4+3*B:4+4*B]
        init_translations += lr * update[4+4*B:4+7*B].reshape(B,3)



       

        loss_stats.append((loss**2).mean().item())
    fx, fy, cx, cy = camera_parameters
    K = torch.stack([fx, torch.tensor(0.0), cx, torch.tensor(0), fy, cy, torch.tensor(0), torch.tensor(0), torch.tensor(1)]).reshape(3,3)
    return K.detach(), loss_stats
