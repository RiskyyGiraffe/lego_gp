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

# Set up the Camera Object
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

# World point tests
# test_array = [[0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]]
#
# draw = ImageDraw.Draw(img)
# size = 7
#
# for test in test_array:
#     u, v, depth = projection(test, camera_object)
#     if depth <= 0:
#         continue
#     x = round(u)
#     y = round(v)
#     if x < 0 or x >= camera_object.width or y < 0 or y >= camera_object.height:
#         continue
#     shape = [x - size, y - size, x + size, y + size]
#     # draw.rectangle(shape, outline="red", width=2)
#
# # img.save("test_draw.png")

def alpha_at_pixel(pixel_x, pixel_y, u, v, sigma_pixels, opacity):
    dx = pixel_x - u
    dy = pixel_y - v
    distance_squared = dx ** 2 + dy ** 2
    weight = np.exp(-0.5 * distance_squared / sigma_pixels ** 2)
    alpha = weight * opacity
    return alpha

img_rgb = img.convert('RGB')
rgb_array = np.array(img_rgb).astype(float) / 255.0

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

color = np.array([1.0, 0.0, 0.0])
# draw_gaussian_splat(rgb_array, 400, 400, 25, color, 0.8)
# img.save("Gaussian_Test.png")

# world_points = [[0, 0, 0.5], [0, 0, 0.0], [0, 0, -0.5]]
# world_radius = 0.1
#
# for wp in world_points:
#     u, v, depth = projection(wp, camera_object)
#     if depth <= 0:
#         continue
#
#     sigma_p = camera_object.fx * world_radius / depth
#     print(depth, sigma_p)
#     draw_gaussian_splat(rgb_array, camera_object.width, camera_object.height, u, v, sigma_p, color, 0.8)
#
# output = Image.fromarray(np.clip(rgb_array * 255, 0, 255).astype(np.uint8))
# output.save("Adjusted_depth.png")

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

gaussians = [Gaussian([0, 0, 0], 0.10, np.array([1.0, 0.0, 0.0]), 0.8),
            Gaussian([0.5, 0, 0], 0.10, np.array([0.0, 1.0, 0.0]), 0.8),
            Gaussian([-0.5, 0, 0], 0.10, np.array([0.0, 0.0, 1.0]), 0.8),
            Gaussian([0, 0.5, 0], 0.10, np.array([1.0, 1.0, 0.0]), 0.8),
            Gaussian([0, 0, 0.5], 0.10, np.array([1.0, 0.0, 1.0]), 0.8)]

rgb_gaus = np.zeros((camera_object.height, camera_object.width, 3))
sort_rgb = []

for gaussian in gaussians:
    wp = gaussian.wp
    radius = gaussian.radius
    color = gaussian.color
    opacity  = gaussian.opacity
    u, v, depth = projection(wp, camera_object)
    if depth <= 0:
        continue

    sigma_p = camera_object.fx * radius / depth
    splat = Splat(camera_object.width, camera_object.height, u, v, depth, sigma_p, gaussian.color, gaussian.opacity)

    sort_rgb.append(splat)

sort_rgb.sort(key=lambda splat: splat.depth, reverse=True)
for d in sort_rgb:
    print(d.depth)

for splat in sort_rgb:
    draw_gaussian_splat(rgb_gaus, splat.width, splat.height, splat.u, splat.v, splat.sigma_p, splat.color, splat.opacity)

output = Image.fromarray(np.clip(rgb_gaus * 255, 0, 255).astype(np.uint8))
output.save("BlankSpaceDepth.png")

