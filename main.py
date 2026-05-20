import json
from PIL import Image
import numpy as np
import os

with open('lego_data/transforms_train.json', 'r') as f:
    loaded_data = json.load(f)

camera_angle_x = loaded_data['camera_angle_x']
frame_fp = loaded_data['frames'][0]['file_path'][2:]
transform_matrix = loaded_data['frames'][0]['transform_matrix']
fp = f"./lego_data/{frame_fp}.png"
img = Image.open(fp)
width, height = img.size
# img.show()
focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

print(f"Width: {width}")
print(f"Height: {height}")
print(f"Camera Angle X: {camera_angle_x}")
print(f"Focal Length: {focal}")
print(f"First Transform Matrix: {transform_matrix}")

directory = "./lego_data/train/"
file_array = [item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))]
file_count = len(file_array)
print(f"Number of training files: {file_count}")
print(f"First Training Item: {frame_fp[6:]}.png")
matrix_array = np.array(transform_matrix)
print(f"Transform Matrix Shape: {matrix_array.shape}")


