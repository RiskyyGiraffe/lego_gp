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
        self.cx, self.cy = 400, 400
        self.c2w = np.array(self.camera_to_world, dtype=np.float64)
        self.w2c = np.linalg.inv(self.c2w)

# Instantiate Camera Object
camera_object = Camera(img, camera_angle_x, camera_to_world)

# Define Projection Function
def projection(p_world_point, camera: Camera):
    wp = np.array(p_world_point)
    world_point = np.append(wp, 1.0)
    p_world = np.array(world_point)
    T_w2c = np.linalg.inv(camera.c2w)
    focal = camera.fx
    p_camera = T_w2c @ p_world
    x_cam, y_cam, z_cam, _ = p_camera
    depth = -z_cam
    u = focal * (x_cam / depth) + camera.width / 2
    v = camera.height / 2 -focal * (y_cam / depth)
    return u, v, depth

# World point tests
test_array = [[0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]]

draw = ImageDraw.Draw(img)
size = 7

for test in test_array:
    u, v, depth = projection(test, camera_object)
    if depth <= 0:
        pass
    x = round(u)
    y = round(v)
    if x > camera_object.width or y > camera_object.height:
        pass
    shape = [x - size, y - size, x + size, y + size]
    draw.rectangle(shape, outline="red", width=2)

# img.save("test_draw.png")

def alpha_at_pixel(pixel_x, pixel_y, u, v, sigma_pixels, opacity):
    dx = pixel_x - u
    dy = pixel_y - v
    distance_squared = dx ** 2 + dy ** 2
    weight = np.exp(-0.5 * distance_squared / sigma_pixels ** 2)
    alpha = weight * opacity
    return alpha

print(alpha_at_pixel(475, 400, 400, 400, 25, 0.8))

img_rgb = img.convert('RGB')
rgb_array = np.array(img_rgb)
frgb_array = np.array()

print(rgb_array[10, 10])

