import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
# 1. 自定义数据集类，加载 JPG 图片
class RGBImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith("-rgb-img.jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     image = self.transform(image)
        return img_name  # 返回图像和文件名

# 2. 数据变换与 DataLoader 设置
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640))  # 根据模型输入尺寸调整
])

data_dir = "/data2/yuhao/class/CS7303/data/cleargrasp-dataset-test-val/real-test/d415"
save_dir = "/data2/yuhao/class/CS7303/data/test"
os.makedirs(save_dir, exist_ok=True)

# 创建数据集与 DataLoader
dataset = RGBImageDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

# 3. 加载 Depth Anything v2 模型
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/data2/yuhao/class/CS7303/pytorch_networks/surface_normals/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
model = model.to(device).eval()  # 假设模型输出深度图

# 4. 生成深度图并保存热力图
def save_heatmap(depth_map, save_path):
    plt.figure()
    plt.imshow(depth_map, cmap='hot')  # 使用热力图颜色映射
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 推理并保存结果
with torch.no_grad():
    for batch_image_names in tqdm(dataloader, desc="Processing Batches"):
        # batch_images = batch_images.to(device)
        # print(batch_images[0].shape)
        # depth_maps = model.infer_image(batch_images[0])
        # depth_maps = depth_maps.cpu().numpy()
        # print(depth_maps.shape)
        raw_images = []
        for name in batch_image_names:
            raw_images.append(cv2.imread(name))
        depth_maps = model.infer_images(raw_images)
        depth_maps = torch.stack([torch.from_numpy(img) for img in depth_maps])
        print(depth_maps.shape)
        # save_path = os.path.join(save_dir, "1")
        # save_heatmap(depth_maps, save_path)
        # 遍历每张图片，保存热力图

        break

print(f"处理完成，结果已保存至 {save_dir}")
