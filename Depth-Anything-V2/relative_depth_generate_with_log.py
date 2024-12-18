import os
from PIL import Image
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import numpy as np
import matplotlib
from tqdm import tqdm
import wandb
import time

# Initialize WandB
wandb.init(project="depth-map-generation")

torch.cuda.empty_cache()
device = 'cuda:1'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
grayscale = False

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/data2/yuhao/class/CS7303/pytorch_networks/surface_normals/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
model = model.to(device).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# List of directories
directories = [
    'data/cleargrasp-dataset-train/stemless-plastic-champagne-glass-train/rgb-imgs',
    'data/cleargrasp-dataset-train/square-plastic-bottle-train/rgb-imgs',
    'data/cleargrasp-dataset-train/heart-bath-bomb-train/rgb-imgs',
    'data/cleargrasp-dataset-train/flower-bath-bomb-train/rgb-imgs',
    'data/cleargrasp-dataset-train/cup-with-waves-train/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-val/stemless-plastic-champagne-glass-val/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-val/square-plastic-bottle-val/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-val/heart-bath-bomb-val/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-val/flower-bath-bomb-val/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-val/cup-with-waves-val/rgb-imgs',
    'data/cleargrasp-dataset-test-val/real-test/d415/',
    'data/cleargrasp-dataset-test-val/real-test/d435/',
    'data/cleargrasp-dataset-test-val/real-val/d435/',
    'data/cleargrasp-dataset-test-val/synthetic-test/glass-round-potion-test/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-test/glass-square-potion-test/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-test/star-bath-bomb-test/rgb-imgs',
    'data/cleargrasp-dataset-test-val/synthetic-test/tree-bath-bomb-test/rgb-imgs',
]

# File suffixes to search for
file_suffixes = ('-transparent-rgb-img.jpg', '-rgb.jpg', '-input-img.jpg')
project_dir = '/data2/yuhao/class/CS7303/'

# Metrics tracking
total_files = 0
processed_files = 0
start_time = time.time()

# Collect all files
all_files = []
for directory in directories:
    directory = os.path.join(project_dir, directory)
    if os.path.exists(directory):  # Check if directory exists
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(file_suffixes):
                    all_files.append(os.path.join(root, file))

total_files = len(all_files)

# Process matching files with tqdm progress bar
for file_path in tqdm(all_files, desc="Processing files"):
    try:
        # Open the image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load {file_path}")
            continue

        # Generate depth map
        depth = model.infer_image(img)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        gray_depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        heat_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Generate the new file name
        base_name = os.path.splitext(file_path)[0]  # Remove the original suffix
        gray_depth_name = os.path.join(os.path.dirname(file_path), base_name + '-relative-depth.jpg')
        heat_depth_name = os.path.join(os.path.dirname(file_path), base_name + '-relative-depth-heatmap.jpg')
        # Save the depth map
        cv2.imwrite(gray_depth_name, gray_depth)
        cv2.imwrite(heat_depth_name, heat_depth)
        processed_files += 1
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Calculate total time and average time per file
end_time = time.time()
total_time = end_time - start_time
average_time_per_file = total_time / max(processed_files, 1)

# Log metrics to WandB
wandb.log({
    "total_files": total_files,
    "processed_files": processed_files,
    "total_time": total_time,
    "average_time_per_file": average_time_per_file
})

print("Processing complete.")
print(f"Total files: {total_files}")
print(f"Successfully processed: {processed_files}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per file: {average_time_per_file:.2f} seconds")
