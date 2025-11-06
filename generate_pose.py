import os
import json
import numpy as np
from PIL import Image
import openpifpaf
from openpifpaf import show

# Paths
input_folder = './datasets/custom/test/person'
output_img_folder = './datasets/custom/test/openpose-img'
output_json_folder = './datasets/custom/test/openpose-json'

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

# Predictor
predictor = openpifpaf.Predictor(checkpoint='resnet50')

# COCO skeleton
skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13],
    [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4],
    [3, 5], [4, 6], [5, 7]
]

for file in os.listdir(input_folder):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_path = os.path.join(input_folder, file)
    base_name = os.path.splitext(file)[0]
    print(f"Processing: {file}")

    image = Image.open(input_path).convert('RGB')
    predictions, _, _ = predictor.numpy_image(np.array(image))

    people_list = []
    keypoints_for_painter = []

    for ann in predictions:
        # Ensure 'data' exists and is non-empty
        data = getattr(ann, 'data', None)
        if data is None or len(data) == 0:
            continue

        # Convert to NumPy array safely
        kps = np.array(data, dtype=np.float32)
        # Reshape if flat list
        if kps.ndim == 1 and len(kps) % 3 == 0:
            kps = kps.reshape(-1, 3)
        else:
            print(f"⚠️ Skipping invalid keypoints in {file}")
            continue

        keypoints_for_painter.append(kps)
        people_list.append({'pose_keypoints_2d': kps.flatten().tolist()})

    # Save JSON
    json_path = os.path.join(output_json_folder, f"{base_name}_keypoints.json")
    with open(json_path, 'w') as f:
        json.dump({'people': people_list}, f)
    print(f"✅ JSON saved: {json_path}")

    # Draw pose if keypoints exist
    if keypoints_for_painter:
        with show.image_canvas(image) as ax:
            painter = show.KeypointPainter(show_box=False)
            for kps in keypoints_for_painter:
                # ✅ Ensure kps is 2D with 3 columns
                if kps.ndim == 2 and kps.shape[1] == 3:
                    painter.keypoints(ax, kps, skeleton=skeleton)

            rendered_path = os.path.join(output_img_folder, f"{base_name}_rendered.png")
            ax.figure.savefig(rendered_path, bbox_inches='tight', pad_inches=0)
            print(f"✅ Pose image saved: {rendered_path}")
    else:
        print(f"⚠️ No keypoints detected in {file}")

print("🎯 Pose generation complete!")
