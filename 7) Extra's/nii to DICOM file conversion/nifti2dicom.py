import SimpleITK as sitk
import os
import time
from glob import glob

def writeSlices(series_tag_values, new_img, i, out_dir):
    image_slice = new_img[:,:,i]
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    filename = 'slice' + str(i).zfill(4) + '.dcm'
    writer.SetFileName(os.path.join(out_dir, filename))
    writer.Execute(image_slice)
    print(f"Slice {filename} written.")

def nifti2dicom_1file(in_dir, out_dir):
    """
    This function is to convert only one nifti file into dicom series

    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    os.makedirs(out_dir, exist_ok=True)

    new_img = sitk.ReadImage(in_dir) 
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    direction = new_img.GetDirection()
    series_tag_values = [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", "Created-Pycad")] # Series Description

    # Write slices to output directory
    print("Writing slices...")
    for i in range(new_img.GetDepth()):
        writeSlices(series_tag_values, new_img, i, out_dir)
    print("Slices writing completed.")

def nifti2dicom_mfiles(nifti_dir, out_dir):
    """
    This function is to convert multiple nifti files into dicom files

    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.

    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    images = glob(os.path.join(nifti_dir, '*.nii.gz'))

    for image in images:
        o_path = os.path.join(out_dir, os.path.basename(image)[:-7])
        os.makedirs(o_path, exist_ok=True)
        print(f"Converting {image} to DICOM...")
        nifti2dicom_1file(image, o_path)
        print(f"Conversion of {image} completed.")

# Paths to directories
nifti_dir = r'G:\FYP\Brats Dataset Training and Validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001'
out_dir = r'G:\FYP\Dicom_files'

# Convert NIfTI to DICOM
print("Starting conversion process...")
nifti2dicom_mfiles(nifti_dir, out_dir)
print("Conversion process completed.")
