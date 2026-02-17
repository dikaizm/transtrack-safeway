import os
import sys

def clean_empty_labels(dataset_dir):
    """Remove empty label files and their corresponding images."""
    
    removed_count = 0
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        images_dir = os.path.join(dataset_dir, split, 'images')
        
        if not os.path.exists(labels_dir):
            print(f"Skipping {split}: labels directory not found")
            continue
            
        print(f"\nChecking {split} split...")
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            
            # Check if file is empty
            if os.path.getsize(label_path) == 0:
                # Remove the label file
                os.remove(label_path)
                print(f"  Removed empty label: {label_file}")
                
                # Remove corresponding image file
                base_name = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"  Removed image: {base_name}{ext}")
                        break
                
                removed_count += 1
    
    print(f"\n✓ Cleaned {removed_count} empty label files and their images")
    return removed_count

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dataset_dir = os.path.join(project_root, 'data', 'road-damage-detection-yolov8-merged')
    
    print(f"Cleaning dataset at: {dataset_dir}")
    clean_empty_labels(dataset_dir)
    print("\nDataset cleaned successfully!")
