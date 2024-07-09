import os
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import imageio
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

##########################
#This part of the code to get an initial understanding of the dataset.
#################################

#PART 1: Load sample images and visualize
#Includes, dividing each image by its max to scale them to [0,1]
#Converting mask from float to uint8
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize
###########################################

#View a few images

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.

BASE_PATH = 'E:/Brats Dataset/BraTS2020_TrainingData/'
TRAIN_DATASET_PATH = os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/')

# Construct the full path to the flair image
# flair_image_path = os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_flair.nii')

# Load the flair image
# test_image_flair = nib.load(flair_image_path).get_fdata()

# Print the maximum value in the image
# rint(test_image_flair.max())

test_image_flair = nib.load(os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_flair.nii')).get_fdata()
print(test_image_flair.max())

# Scalers are applied to 1D so let us reshape and then reshape back to original shape.
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

test_image_t1 = nib.load(os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1.nii')).get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce = nib.load(os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1ce.nii')).get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2 = nib.load(os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t2.nii')).get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask = nib.load(os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_seg.nii')).get_fdata()
test_mask = test_mask.astype(np.uint8) #used to convert from floating point to unsigned integers

print(np.unique(test_mask))  # 0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3) LABELS used in the image 
test_mask[test_mask == 4] = 3  # Reassign mask values 4 to 3
print(np.unique(test_mask))

import random
n_slice = random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

##################################################
#PART 2: Explore the process of combining images to channels and divide them to patches
#Includes...
#Combining all 4 images to 4 channels of a numpy array.
#
################################################
# Flair, T1CE, and T2 have the most information
# Combine t1ce, t2, and flair into single multichannel image

combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)

# Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
# cropping x, y, and z
combined_x = combined_x[56:184, 56:184, 13:141]  # Crop to 128x128x128x4

# Do the same for mask
test_mask = test_mask[56:184, 56:184, 13:141]

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

imsave('E:/Brats Dataset/BraTS2020_TrainingData/combined255.tif', combined_x) # Can save multi-channel files at most 3 channels
np.save('E:/Brats Dataset/BraTS2020_TrainingData/combined255.npy', combined_x) # saves data as a numpy array. used so that we can store images with more than 3 channels

# Verify image is being read properly
# my_img=imread('BraTS2020_TrainingData/combined255.tif')

my_img = np.load('E:/Brats Dataset/BraTS2020_TrainingData/combined255.npy')

test_mask = to_categorical(test_mask, num_classes=4) # classes are 0,1,2,3

##################################################
#End of understanding the dataset. Now get it organized.
#####################################

# Now let us apply the same as above to all the images...
# Merge channels, crop, patchify, save
# GET DATA READY =  GENERATORS OR OTHERWISE

# Keras datagenerator does not support 3d

# # # images lists harley
t1_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t1.nii')))
t2_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*flair.nii')))
mask_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*seg.nii')))
# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes


#numpy array type file storage


# Define base path
BASE_PATH = r'E:\Brats Dataset\BraTS2020_TrainingData'

# Define paths for saving images and masks
IMAGES_PATH = os.path.join(BASE_PATH, 'input_data_3channel_npy', 'images')
MASKS_PATH = os.path.join(BASE_PATH, 'input_data_3channel_npy', 'masks')

# Create directories if they do not exist
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(MASKS_PATH, exist_ok=True)

# Load the list of images
t1_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t1.nii')))
t2_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*flair.nii')))
mask_list = sorted(glob.glob(os.path.join(BASE_PATH, 'MICCAI_BraTS2020_TrainingData/*/*seg.nii')))

# Process each image
for img in range(len(t2_list)):   # Using t1_list as all lists are of the same size
    print("Now preparing image and masks number: ", img)
    
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # cropping x, y, and z
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    # Increment file names for saving
    filename = 'image_' + str(img) + '.npy'
    maskname = 'mask_' + str(img) + '.npy'
    
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        try:
            np.save(os.path.join(IMAGES_PATH, filename), temp_combined_images)
            np.save(os.path.join(MASKS_PATH, maskname), temp_mask)
        except Exception as e:
            print(f"Error occurred while saving files at index {img}: {e}")
    else:
        print("I am useless")



# #tiff file storage


# # Define base path

# BASE_PATH = r'E:\Brats Dataset\BraTS2020_TrainingData'

# # Define paths for saving images and masks
# IMAGES_PATH = os.path.join(BASE_PATH, 'input_data_3channel_tiff', 'images')
# MASKS_PATH = os.path.join(BASE_PATH, 'input_data_3channel_tiff', 'masks')

# for img in range(len(t2_list)):   # Using t1_list as all lists are of the same size
#     print("Now preparing image and masks number: ", img)
    
#     temp_image_t2 = nib.load(t2_list[img]).get_fdata()
#     temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
#     temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
#     temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
#     temp_image_flair = nib.load(flair_list[img]).get_fdata()
#     temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
#     temp_mask = nib.load(mask_list[img]).get_fdata()
#     temp_mask = temp_mask.astype(np.uint8)
#     temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
    
#     temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
#     # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
#     # cropping x, y, and z
#     temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
#     temp_mask = temp_mask[56:184, 56:184, 13:141]
    
#     val, counts = np.unique(temp_mask, return_counts=True)
    
#     # Increment file names for saving
#     filename_prefix = 'image_' + str(img)
#     maskname_prefix = 'mask_' + str(img)
    
#     if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
#         print("Save Me")
#         temp_mask = to_categorical(temp_mask, num_classes=4)
#         try:
#             for i in range(temp_combined_images.shape[2]):
#                 imageio.imwrite(os.path.join(IMAGES_PATH, f"{filename_prefix}_{i}.tiff"), temp_combined_images[:, :, i])
#                 imageio.imwrite(os.path.join(MASKS_PATH, f"{maskname_prefix}_{i}.tiff"), temp_mask[:, :, i])
#         except Exception as e:
#             print(f"Error occurred while saving files at index {img}: {e}")
#     else:
#         print("I am useless")



# # Define base path
BASE_PATH = r'E:\Brats Dataset\BraTS2020_TrainingData'



# Repeat the same from above for validation data folder OR
# Split training data into train and validation

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators.

pip install split-folders
"""
import splitfolders  # or import split_folders

input_folder = os.path.join(BASE_PATH, 'input_data_3channel_npy/')
output_folder = os.path.join(BASE_PATH, 'input_data_128/')
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)  # default values
