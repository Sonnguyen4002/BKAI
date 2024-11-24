import torch
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from model import model1

# Khởi tạo thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Hàm tiền xử lý ảnh
def preprocess_image(image_path, target_size=(256, 256)):
    # Đọc và xử lý ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be opened: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, target_size)

    # Áp dụng transform giống trong UNetDataset
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    processed = transform(image=resized_image)
    return processed["image"].unsqueeze(0), image  # Trả về tensor và ảnh gốc


# Hàm lưu ảnh dự đoán phân đoạn
def save_predicted_segmentation(predicted_mask, output_path="predict_image.png"):
    plt.figure(figsize=(6, 6))
    plt.imshow(predicted_mask, cmap="viridis")  # Hiển thị mask dự đoán với cmap
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Predicted segmentation saved as {output_path}")


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, help="Path to the input image")
args = parser.parse_args()
image_path = args.image_path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

input_tensor, original_image = preprocess_image(image_path)
input_tensor = input_tensor.to(device)

model = model1.to(device)
checkpoint = torch.load('../../../old_shit/model.pth', map_location=device)
model.load_state_dict(checkpoint['model'])

model.eval()

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

predicted_mask = cv2.resize(prediction, (original_image.shape[1], original_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

save_predicted_segmentation(predicted_mask)
