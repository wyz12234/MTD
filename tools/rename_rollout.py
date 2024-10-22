import os
import time

def check_and_rename(file_dir, target_file):
    while True:
        files = os.listdir(file_dir)

        if target_file in files:
            base_name, ext = os.path.splitext(target_file)
            i = 1
            while True:
                new_name = f"{base_name}_{i}{ext}"
                if new_name not in files:
                    break
                i += 1
            
            old_path = os.path.join(file_dir, target_file)
            new_path = os.path.join(file_dir, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed '{target_file}' to '{new_name}'")
        
        time.sleep(10)
        
file_dir = '/home/wyz/diffusion_trajection/nusc_results_scene_0098/query_edit_eval/viz'
target_file = 'scene-0276_0031.mp4'

check_and_rename(file_dir, target_file)