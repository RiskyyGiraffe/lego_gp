import json
from PIL import Image, ImageDraw
import numpy as np
import os
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Define the Class Objects
class CameraTorch:
    def __init__(self, img, camera_angle_x, camera_to_world, scale=1.0):
        if scale != 1.0:
            new_width = int(img.size[0] * scale)
            new_height = int(img.size[1] * scale)
            img = img.resize((new_width, new_height))

        self.img = img
        self.width, self.height = img.size

        self.camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32)
        self.fx = 0.5 * self.width / torch.tan(torch.tensor(0.5) * camera_angle_x)
        self.fy = self.fx
        self.cx, self.cy = self.width / 2, self.height / 2
        self.c2w = torch.tensor(camera_to_world, dtype=torch.float32)
        self.w2c = torch.linalg.inv(self.c2w)

class GaussianCloud:
    def __init__(self, init_positions, init_radius=0.10, init_opacity=0.8):
        self.positions = torch.tensor(init_positions, dtype=torch.float32, requires_grad=True)
        self.colors_raw = torch.zeros((len(init_positions), 3), dtype=torch.float32, requires_grad=True)
        self.radii_raw = torch.full(
            (len(init_positions),),
            float(np.log(init_radius)),
            dtype=torch.float32,
            requires_grad=True
        )
        init_opacity_raw = np.log(init_opacity / (1 - init_opacity))
        self.opacity_raw = torch.full(
            (len(init_positions),),
            float(init_opacity_raw),
            dtype=torch.float32,
            requires_grad=True)

    def parameters(self):
        return [self.positions, self.colors_raw, self.radii_raw, self.opacity_raw]

# Hyperparams
N = 100
views = 5
scale_factor = 0.5
init_radius = 0.05
lr = 0.01
steps = 1001

# Load the data
with open('lego_data/transforms_train.json', 'r') as f:
    loaded_data = json.load(f)
camera_angle_x = loaded_data['camera_angle_x']

training_items = []

for frame in loaded_data['frames'][:views]:
    frame_fp = frame['file_path'][2:]
    # camera_angle_x = frame['camera_angle_x']
    camera_to_world = frame['transform_matrix']
    fp = f"./lego_data/{frame_fp}.png"

    img = Image.open(fp).convert("RGB")
    camera = CameraTorch(img, camera_angle_x, camera_to_world, scale_factor)

    target_np = np.array(camera.img).astype(float) / 255.0
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    training_items.append((camera, target_torch))

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
    opacity = torch.sigmoid(cloud.opacity_raw[i])

    ys = torch.arange(0, camera.height, dtype=torch.float32)
    xs = torch.arange(0, camera.width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    dx = xx - u
    dy = yy - v

    dist2 = dx ** 2 + dy ** 2
    weight = torch.exp(-0.5 * dist2 / (sigma_p ** 2))
    alpha = opacity * weight

    splat_img = alpha[..., None] * color[None, None, :]

    return alpha, color


def torch_scene_cloud(camera, cloud):
    image = torch.zeros((camera.height, camera.width, 3), dtype=torch.float32)
    T = torch.ones((camera.height, camera.width, 1), dtype=torch.float32)

    N = cloud.positions.shape[0]

    for i in range(N):
        alpha, color = torch_render_cloud_splat(camera, cloud, i)
        alpha_3 = alpha[..., None]
        image = image + T * alpha_3 * color[None, None, :]
        T = T * (1 - alpha_3)

    return image

torch_camera_object = CameraTorch(img, camera_angle_x, camera_to_world)

init_positions = torch.randn((N, 3)) * 0.35
init_positions[:, 2] *= 0.5
init_positions = init_positions.tolist()

def projection_torch(p_world_point, camera: CameraTorch):
    one = torch.ones(1, dtype=p_world_point.dtype)
    p_world = torch.cat([p_world_point, one], dim=0)
    p_camera = camera.w2c @ p_world
    x_cam, y_cam, z_cam, _ = p_camera
    depth = -z_cam
    u = camera.fx * (x_cam / depth) + camera.cx
    v = camera.cy - camera.fy * (y_cam / depth)
    return u, v, depth

cloud = GaussianCloud(init_positions, init_radius=init_radius)
optimizer = torch.optim.Adam(cloud.parameters(), lr=lr)

img_rgb = img.convert("RGB")
target = np.array(img_rgb).astype(float) / 255.0
target_torch = torch.tensor(target, dtype=torch.float32)

for step in range(steps):
    camera, target_torch = training_items[step % len(training_items)]

    predicted = torch_scene_cloud(camera, cloud)
    loss = torch.mean(torch.abs(predicted - target_torch))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(step, loss.item())
        print("positions:", cloud.positions.detach())
        print("colors:", torch.sigmoid(cloud.colors_raw).detach())
        print("radii:", torch.exp(cloud.radii_raw).detach())
        print("opacities:", torch.sigmoid(cloud.opacity_raw).detach())

        with torch.no_grad():
            losses = []
            for camera, target in training_items:
                pred = torch_scene_cloud(camera, cloud)
                losses.append(torch.mean(torch.abs(pred - target)).item())

            print("avg loss:", sum(losses) / len(losses))

        with torch.no_grad():
            for idx, (camera, target) in enumerate(training_items[:3]):
                pred = torch_scene_cloud(camera, cloud)
                output = Image.fromarray(
                    np.clip(pred.detach().numpy() * 255, 0, 255).astype(np.uint8)
                )
                output.save(f"multi_view_render_{idx}.png")
