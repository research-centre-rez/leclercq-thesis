import os
import masking
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='../../data', type=str, help='Root directory containing the images to be cropped')
parser.add_argument('--model', default='leclercq', type=str, choices=['leclercq', 'sirotenko'], help='Choose which model to use: [sirotenko, leclercq]')
parser.add_argument('--crop_size', default=2200, type=int, help='Size of the cropped image')
parser.add_argument('--model_weights', default='../../weights/concrete_segmentation_unet/unet_efficientnet-b6', type=str, help='Path to model')
parser.add_argument('--encoder_used', default='efficientnet-b6', help='Name of encoder used')

def crop_images(root_dir:str, model_choice:str, crop_size:int, weights_path:str, encoder_name:str) -> None:
    """
    Given a root directory that contains the images, walk through it and only crop images that need to be cropped.
    """
    model = None

    if model_choice == 'leclercq':
        model = masking.my_smp_model(weights_path, encoder_name)
    elif model_choice == 'sirotenko':
        model = masking.sirotenko_model()

    for dir_, _, files in os.walk(root_dir, topdown=True):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, os.getcwd())
            rel_file = os.path.join(rel_dir, file_name)

            if file_name.endswith('.jpg') and "cropped" not in file_name:
                print(f"Cropping {file_name}")
                if model_choice == 'leclercq':
                    masking.my_segmentation(model, rel_file, rel_dir, crop_size)
                elif model_choice == 'sirotenko':
                    masking.sirotenko_segmentation(model, rel_file, rel_dir, crop_size)

def main(args:argparse.Namespace) -> None:
    crop_images(args.root_dir, args.model, args.crop_size, args.model_weights, args.encoder_used)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

