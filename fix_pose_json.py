import os, json

# Path to your generated pose JSONs
input_folder = './datasets/custom/test/openpose-json'

for file in os.listdir(input_folder):
    if file.endswith('_keypoints.json'):
        path = os.path.join(input_folder, file)

        with open(path, 'r') as f:
            data = json.load(f)

        # Skip already-correct files
        if isinstance(data, dict) and 'people' in data:
            continue

        # Convert from OpenPifPaf to VITON-HD compatible format
        if isinstance(data, list) and len(data) > 0 and 'keypoints' in data[0]:
            formatted = {
                "people": [
                    {"pose_keypoints_2d": data[0]["keypoints"]}
                ]
            }

            with open(path, 'w') as f:
                json.dump(formatted, f)
            print(f"✅ Converted: {file}")

print("🎯 All pose JSON files converted successfully!")
