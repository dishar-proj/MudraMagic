import os
import shutil
import random

# This script is used to ensure the dataset used for training is balanced.

# This function determines which image folder has the fewest images
# and randomly pulls images from the folders with a greater number of images
# and puts them into test_balanced and train_balanced. 
def balance_dataset(source_dir, balanced_dir):
    print(f"\n--- Balancing {source_dir} ---")
    
    #perform a check to see if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Could not find the folder '{source_dir}'")
        return

    #get list of all the mudra subfolders
    mudras = [m for m in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, m))]
    
    if not mudras: # throws an error message if no subfolders are found
        print(f"Error: No subfolders found inside '{source_dir}'")
        return

    # 1) Find the minimum number of images across all folders
    min_images = float('inf')
    for mudra in mudras:
        mudra_path = os.path.join(source_dir, mudra)
        num_images = len(os.listdir(mudra_path))
        print(f"  - {mudra} has {num_images} images.")
        
        if num_images < min_images:
            min_images = num_images
    # print statement for debugging
    print(f"\n> Lowest count is {min_images}. Balancing all classes to {min_images} images...\n")

    # 2) Randomly copy the lowest number of images to the balanced directory
    for mudra in mudras:
        source_mudra_path = os.path.join(source_dir, mudra)
        dest_mudra_path = os.path.join(balanced_dir, mudra)
        
        #create the new subfolder for this mudra
        os.makedirs(dest_mudra_path, exist_ok=True)
        
        # get all images in the mudra's folder
        all_images = os.listdir(source_mudra_path)
        
        # randomize to get a random selection
        random.shuffle(all_images)
        
        # again select min number of images
        selected_images = all_images[:min_images]
        
        #Copy the images to the new balanced folder
        for img in selected_images:
            source_file = os.path.join(source_mudra_path, img)
            dest_file = os.path.join(dest_mudra_path, img)
            shutil.copy(source_file, dest_file)
            
        print(f"  Copied {min_images} random images to {dest_mudra_path}")
    print(f"--- Finished balancing {source_dir}! ---\n")


#MAIN: call the function on testing and training sets 
#Training Data balancing
balance_dataset(source_dir='dataset/train', balanced_dir='dataset/train_balanced')

#Testing Data balancing
balance_dataset(source_dir='dataset/test', balanced_dir='dataset/test_balanced')

print("Balancing Complete.")