import os
import cv2
import numpy as np

input_folder = './datasets/custom/test/cloth'
output_folder = './datasets/custom/test/cloth-mask'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {filename}, could not load.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to create binary mask
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Optional: small cleaning to smooth edges
        mask = cv2.medianBlur(mask, 5)

        mask_path = os.path.join(output_folder, filename)
        cv2.imwrite(mask_path, mask)
        print(f"Generated mask for: {filename}")

print("✅ All cloth masks generated successfully!")
