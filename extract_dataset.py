import os
from tqdm import tqdm
from PIL import Image

def main():
    data_dis = [
        # "/home/pushkalm11/Courses/ece285/Project/dataset/s1_2025-03-05",
        "/home/pushkalm11/Courses/ece285/Project/dataset/s3_2025-03-05"
    ]
    save_path = "/home/pushkalm11/Courses/ece285/Project/dataset/images"
    save_num = 35333
    for scenario_dir in data_dis:
        town_paths = [os.path.join(scenario_dir, town) for town in os.listdir(scenario_dir)]
        for i, town_path in enumerate(town_paths):
            route_paths = [os.path.join(town_path, route) for route in os.listdir(town_path)]
            # ignore .json or .sh files
            route_paths = [path for path in route_paths if not path.endswith(".json") and not path.endswith(".sh")]
            for j, route_path in enumerate(route_paths):
                
                image_dir = os.path.join(route_path, "rgb")
                # check if image_dir exists else skip
                if not os.path.exists(image_dir):
                    print(f"Image directory {image_dir} does not exist")
                    continue
                image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
                
                pbar = tqdm(
                    total = len(image_files), 
                    desc = f"Processing Town {i + 1} / {len(town_paths)} - Route {j + 1} / {len(route_paths)}",
                    position = 0,
                    leave = True
                )
                
                for image_file in image_files:
                    image = Image.open(image_file)
                    image.save(os.path.join(save_path, f"{save_num:06d}.png"))
                    save_num += 1
                    pbar.update(1)
                pbar.close()

if __name__ == "__main__":
    main()