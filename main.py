import json
from PIL import Image, ImageDraw
import numpy as np
import os

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
class Camera:
    def __init__(self, img, camera_angle_x, camera_to_world):
        self.img = img
        self.camera_angle_x = camera_angle_x
        self.camera_to_world = camera_to_world
        self.width, self.height = img.size
        self.fx = 0.5 * self.width / np.tan(0.5 * camera_angle_x)
        self.fy = self.fx
        self.cx, self.cy = self.width / 2, self.height / 2
        self.c2w = np.array(self.camera_to_world, dtype=np.float64)
        self.w2c = np.linalg.inv(self.c2w)

class Gaussian:
    def __init__ (self, wp, radius, color, opacity):
        self.wp = wp
        self.radius = radius
        self.color = color
        self.opacity = opacity

class Splat:
    def __init__ (self, width, height, u, v, depth, sigma_p, color, opacity):
        self.width = width
        self.height = height
        self.u = u
        self.v = v
        self.depth = depth
        self.sigma_p = sigma_p
        self.color = color
        self.opacity = opacity

# Instantiate Camera Object
camera_object = Camera(img, camera_angle_x, camera_to_world)

# Define Projection Function
def projection(p_world_point, camera: Camera):
    wp = np.array(p_world_point)
    world_point = np.append(wp, 1.0)
    p_world = np.array(world_point)
    focal = camera.fx
    p_camera = camera.w2c @ p_world
    x_cam, y_cam, z_cam, _ = p_camera
    depth = -z_cam
    u = focal * (x_cam / depth) + camera.width / 2
    v = camera.height / 2 -focal * (y_cam / depth)
    return u, v, depth

# Define Alpha Function
def alpha_at_pixel(pixel_x, pixel_y, u, v, sigma_pixels, opacity):
    dx = pixel_x - u
    dy = pixel_y - v
    distance_squared = dx ** 2 + dy ** 2
    weight = np.exp(-0.5 * distance_squared / sigma_pixels ** 2)
    alpha = weight * opacity
    return alpha

# Previously converting image to rgb array
# img_rgb = img.convert('RGB')
# rgb_array = np.array(img_rgb).astype(float) / 255.0

# Define Draw Function
def draw_gaussian_splat(pixel_array, width, height, u, v, sigma_pixels, color, opacity):
    radius = 3 * sigma_pixels
    x_min = max(0, int(u - radius))
    x_max = min(width - 1, int(u + radius))

    y_min = max(0, int(v - radius))
    y_max = min(height - 1, int(v + radius))
    # pixels = img.load()
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            old_pixel = pixel_array[y, x]
            alpha = alpha_at_pixel(x, y, u, v, sigma_pixels, opacity)
            # new_pixel_x, new_pixel_y, new_pixel_z = (old_pixel * (1 - alpha) + color * alpha) * 255.0
            new_pixel = old_pixel * (1 - alpha) + color * alpha
            # rgb_array[y, x] = (int(new_pixel_x), int(new_pixel_y), int(new_pixel_z))
            pixel_array[y, x] = new_pixel

# Define Render Function
def render_scene(camera: Camera, gaussians: list[Gaussian]):

    rgb_gaus = np.zeros((camera.height, camera.width, 3))
    sort_rgb = []

    for gaussian in gaussians:
        wp = gaussian.wp
        radius = gaussian.radius
        color = gaussian.color
        opacity  = gaussian.opacity
        u, v, depth = projection(wp, camera)

        if depth <= 0:
            continue

        sigma_p = camera.fx * radius / depth
        splat = Splat(camera.width, camera.height, u, v, depth, sigma_p, gaussian.color, gaussian.opacity)

        sort_rgb.append(splat)

    sort_rgb.sort(key=lambda splat: splat.depth, reverse=True)

    for splat in sort_rgb:
        draw_gaussian_splat(rgb_gaus, splat.width, splat.height, splat.u, splat.v, splat.sigma_p, splat.color, splat.opacity)

    return rgb_gaus

gaussians_tests =  [Gaussian([0, 0, 0], 0.10, np.array([1.0, 0.0, 0.0]), 0.8),
                    Gaussian([0.5, 0, 0], 0.10, np.array([0.0, 1.0, 0.0]), 0.8),
                    Gaussian([-0.5, 0, 0], 0.10, np.array([0.0, 0.0, 1.0]), 0.8),
                    Gaussian([0, 0.5, 0], 0.10, np.array([1.0, 1.0, 0.0]), 0.8),
                    Gaussian([0, 0, 0.5], 0.10, np.array([1.0, 0.0, 1.0]), 0.8)]

pixel_array = render_scene(camera_object, gaussians_tests)

output = Image.fromarray(np.clip(pixel_array * 255, 0, 255).astype(np.uint8))
output.save("BlankSpaceFunc.png")

