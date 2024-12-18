import os
from PIL import Image
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import numpy as np
import matplotlib

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

# Process matching files
for directory in directories:
    directory = os.path.join(project_dir, directory)
    if os.path.exists(directory):  # Check if directory exists
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(file_suffixes):
                    file_path = os.path.join(root, file)
                    
                    # Open the image
                    img = cv2.imread(file_path)
                    # 生成深度图    
                    depth = model.infer_image(img)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = depth.astype(np.uint8)
                    
                    if grayscale:
                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                    else:
                        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    
                    # if args.pred_only:
                    #     cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
                    # else:
                    #     split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                    #     combined_result = cv2.hconcat([raw_image, split_region, depth])
                        
                    base_name = os.path.splitext(file_path)[0]  # Remove the original suffix
                    new_file_name = os.path.join(root, base_name  + '-relative-depth.jpg')
                    # 存储深度图
                    cv2.imwrite(new_file_name, depth)

                    print(f"Processed and saved: {new_file_name}")

print("Processing complete.")
