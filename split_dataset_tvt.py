import os
import shutil
import argparse
from glob import glob

def split_dataset(train_ratio, val_ratio, folder, dest):
    # Create destination directories
    images_dir = os.path.join(dest, 'images')
    labels_dir = os.path.join(dest, 'labels')
    
    for sub_dir in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, sub_dir), exist_ok=True)
    
    # Get all image files
    image_files = glob(os.path.join(folder, '*.jpg'))
    
    # Group images by camera
    camera_images = {}
    for file_path in image_files:
        camera_id = os.path.basename(file_path).split('_')[1]
        if camera_id not in camera_images:
            camera_images[camera_id] = []
        camera_images[camera_id].append(file_path)
    
    # Function to copy files
    def copy_files(file_list, images_dest_dir, labels_dest_dir):
        for file_path in file_list:
            shutil.copy(file_path, images_dest_dir)
            txt_file_path = file_path.replace('.jpg', '.txt')
            if os.path.exists(txt_file_path):
                shutil.copy(txt_file_path, labels_dest_dir)
    
    # Split and copy files for each camera
    for camera_id, files in camera_images.items():
        files.sort()  # Ensure files are sorted by name
        num_files = len(files)
        train_end_idx = int(num_files * (train_ratio / 100))
        val_start_idx = int(num_files * ((100 - val_ratio) / 100))
        
        train_files = files[:train_end_idx]
        val_files = files[val_start_idx:]
        
        copy_files(train_files, os.path.join(images_dir, 'train'), os.path.join(labels_dir, 'train'))
        copy_files(val_files, os.path.join(images_dir, 'val'), os.path.join(labels_dir, 'val'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train and validation sets.')
    parser.add_argument('--train', type=int, required=True, help='Percentage of training data')
    parser.add_argument('--validation', type=int, required=True, help='Percentage of validation data')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the dataset')
    parser.add_argument('--dest', type=str, required=True, help='Destination folder for the split dataset')
    
    args = parser.parse_args()
    
    split_dataset(args.train, args.validation, args.folder, args.dest)