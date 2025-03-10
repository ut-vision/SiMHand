import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Import the EGO4D_DB class from the data_loader module
from src.data_loader.ego4d_loader import EGO4D_DB
# Import the 100DOH_DB class from the data_loader module
from src.data_loader.doh_loader import DOH_DB

#########################################################################
dataset = "EGO4D"  # "EGO4D"
datasets_scale = "50k"
json_path = f'/home/nielin/datasets/Hand100M/annotations/Ego4D/Hand100M_Ego4D_{datasets_scale}_v1-1.json'
data_dir = '/work/nielin/datasets/Hand100M'
split = 'train'
output_dir = './Hand100M/visualization'
vis_indices =  [32793, 5400, 11193,3853,8172, 438, 
                33450, 29931, 26570, 26056,   
                21266, 31699, 45636,48900,26102,
                9433, 37742, 6288,49921,39327,
                33158, 27045, 11754,6008,46605,
                43014, 18308, 21070, 9114,535,
                2904, 9873, 15716,8117,45060,
                39359, 9943, 27840,35185,22536,
                36571, 18042, 41388,38874,24025,
                24395, 25339, 29087,25194,18281,
                ]
##########################################################################################################

db = EGO4D_DB(root_dir=data_dir, 
                split='train', 
                datasets_scale = datasets_scale
                )

os.makedirs(output_dir, exist_ok=True)
print(f"Visulization folder created at: {output_dir}")

# Get the first sample from the dataset
sample = db[0]
img_crop = sample["image"]
img_height, img_width, _ = img_crop.shape

# Define the grid size
grid_size = (10, 10)  # 10x10 grid
grid_height = img_crop.shape[0] * grid_size[0]
grid_width = img_crop.shape[1] * grid_size[1]

# Make a blank image for the grid
grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
pair_size = 2  # Each pair consists of 2 images
pairs_per_row = 5  # 5 pairs per row

# Iterate over the grid
for row in range(grid_size[0]):
    for pair in range(pairs_per_row):
        # Calculate the index of the sample
        idx = vis_indices[row * pairs_per_row + pair]
        
        # Get the sample and the positive sample
        sample = db[idx]
        positive_sample_idx = sample["positive_sample_idx"]
        positive_sample = db[positive_sample_idx]
        
        # Get the images
        sample_img = sample["image"]
        positive_sample_img = positive_sample["image"]
        
        # Calculate the x position
        x_start = pair * pair_size * img_width
        x_end = x_start + img_width
        
        # Calculate the y position
        y_start = row * img_height
        y_end = y_start + img_height
        
        grid_image[y_start:y_end, x_start:x_start + img_width] = sample_img
        grid_image[y_start:y_end, x_start + img_width:x_end + img_width] = positive_sample_img

# Convert the grid image to RGB
grid_image_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)

# Save the grid image
output_path = os.path.join(output_dir, 'vis_10x10_grid.jpg')
cv2.imwrite(output_path, grid_image_rgb)
print(f"Image saved to: {output_path}")