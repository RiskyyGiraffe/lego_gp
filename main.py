import json
from PIL import Image, ImageDraw
import numpy as np
import os
import torch

# Load the data
with open('lego_data/transforms_train.json', 'r') as f:
    loaded_data = json.load(f)
# Create the objects from the data
camera_angle_x = loaded_data['camera_angle_x']
frame_fp = loaded_data['frames'][0]['file_path'][2:]
camera_to_world = loaded_data['frames'][0]['transform_matrix']
fp = f"./lego_data/{frame_fp}.png"
img = Image.open(fp)

# Define the Class Objects
class CameraTorch:
    def __init__(self, img, camera_angle_x, camera_to_world):
        self.img = img
        self.camera_angle_x = torch.tensor(camera_angle_x)
        self.camera_to_world = camera_to_world
        self.width, self.height = img.size
        self.fx = 0.5 * self.width / torch.tan(torch.tensor(0.5) * camera_angle_x)
        self.fy = self.fx
        self.cx, self.cy = self.width / 2, self.height / 2
        self.c2w = torch.tensor(self.camera_to_world, dtype=torch.float32)
        self.w2c = torch.linalg.inv(self.c2w)

class GaussianCloud:
    def __init__(self, init_positions, init_radius=0.10):
        self.positions = torch.tensor(init_positions, dtype=torch.float32, requires_grad=True)
        self.colors_raw = torch.zeros((len(init_positions), 3), dtype=torch.float32, requires_grad=True)
        self.radii_raw = torch.full(
            (len(init_positions),),
            float(np.log(init_radius)),
            dtype=torch.float32,
            requires_grad=True
        )
        self.opacity = 0.8

    def parameters(self):
        return [self.positions, self.colors_raw, self.radii_raw]

def torch_alpha_at_pixel(pixel_x, pixel_y, u, v, sigma_pixels, opacity):
    dx = pixel_x - u
    dy = pixel_y - v
    distance_squared = dx ** 2 + dy ** 2
    weight = torch.exp(-0.5 * distance_squared / sigma_pixels ** 2)
    alpha = weight * opacity
    return alpha

def torch_render_cloud_splat(camera: CameraTorch, cloud: GaussianCloud, i):
    position = cloud.positions[i]
    u, v, depth = projection_torch(position, camera)
    radius = torch.exp(cloud.radii_raw[i])
    sigma_p = camera.fx * radius / depth
    color = torch.sigmoid(cloud.colors_raw[i])
    opacity = cloud.opacity

    ys = torch.arange(0, camera.height, dtype=torch.float32)
    xs = torch.arange(0, camera.width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    dx = xx - u
    dy = yy - v

    dist2 = dx ** 2 + dy ** 2
    weight = torch.exp(-0.5 * dist2 / (sigma_p ** 2))
    alpha = opacity * weight

    splat_img = alpha[..., None] * color[None, None, :]

    return splat_img


def torch_scene_cloud(camera, cloud):
    image = torch.zeros((camera.height, camera.width, 3), dtype=torch.float32)

    N = cloud.positions.shape[0]

    for i in range(N):
        splat_img = torch_render_cloud_splat(camera, cloud, i)
        image = image + splat_img

    image = torch.clamp(image, 0.0, 1.0)
    return image

torch_camera_object = CameraTorch(img, camera_angle_x, camera_to_world)

N = 20
init_positions = torch.randn((N, 3)) * 0.25
init_positions[:, 2] *= 0.25
init_positions = init_positions.tolist()


def projection_torch(p_world_point, camera: CameraTorch):
    one = torch.ones(1, dtype=p_world_point.dtype)
    p_world = torch.cat([p_world_point, one], dim=0)
    focal = camera.fx
    p_camera = camera.w2c @ p_world
    x_cam, y_cam, z_cam, _ = p_camera
    depth = -z_cam
    u = focal * (x_cam / depth) + camera.width / 2
    v = camera.height / 2 -focal * (y_cam / depth)
    return u, v, depth

cloud = GaussianCloud(init_positions)
optimizer = torch.optim.Adam(cloud.parameters(), lr=0.01)

img_rgb = img.convert("RGB")
target = np.array(img_rgb).astype(float) / 255.0
target_torch = torch.tensor(target, dtype=torch.float32)

for step in range(201):
    predicted = torch_scene_cloud(torch_camera_object, cloud)
    loss = torch.mean(torch.abs(predicted - target_torch))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(step, loss.item())
        print("positions:", cloud.positions.detach())
        print("colors:", torch.sigmoid(cloud.colors_raw).detach())
        print("radii:", torch.exp(cloud.radii_raw).detach())

with torch.no_grad():
    predicted = torch_scene_cloud(torch_camera_object, cloud)

output = Image.fromarray(
    np.clip(predicted.detach().numpy() * 255, 0, 255).astype(np.uint8)
)
output.save("cloud_trained_3.png")
