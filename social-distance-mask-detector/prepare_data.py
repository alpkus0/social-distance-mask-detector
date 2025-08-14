import os
import shutil
import random

# Paths to the original image and label directories
images_dir = 'dataset/images'        # Directory containing the original images
labels_dir = 'dataset/labels'        # Directory containing the corresponding labels

# Paths for training and validation image/label directories
train_img_dir = 'dataset/images/train'    # Training images will be moved here
val_img_dir = 'dataset/images/val'        # Validation images will be moved here
train_label_dir = 'dataset/labels/train'  # Training labels will be moved here
val_label_dir = 'dataset/labels/val'      # Validation labels will be moved here

# Create directories if they don't exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# List all image files with supported extensions
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the list for random distribution
random.shuffle(all_images)

# Split: 80% for training, 20% for validation
train_count = int(len(all_images) * 0.8)
train_images = all_images[:train_count]  # First 80% for training
val_images = all_images[train_count:]    # Remaining 20% for validation

# Function to move image and corresponding label files
def move_files(file_list, src_img_dir, dst_img_dir, src_label_dir, dst_label_dir):
    for filename in file_list:
        # Move image to the target directory
        shutil.move(os.path.join(src_img_dir, filename), os.path.join(dst_img_dir, filename))

        # Move corresponding label file
        label_file = filename.rsplit('.', 1)[0] + '.txt'
        shutil.move(os.path.join(src_label_dir, label_file), os.path.join(dst_label_dir, label_file))

# Move training files
move_files(train_images, images_dir, train_img_dir, labels_dir, train_label_dir)

# Move validation files
move_files(val_images, images_dir, val_img_dir, labels_dir, val_label_dir)

print("Dataset successfully split into training and validation sets.")
