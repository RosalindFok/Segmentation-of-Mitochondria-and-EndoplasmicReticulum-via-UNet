"""
config file: load path, set hyperparameters
"""

import os
import re

''' Load Path '''
data_dir_path = os.path.join(os.getcwd(), '..', 'data')
ER_dataset_dir_path = os.path.join(data_dir_path, 'ER_dataset')
MITO_dataset_dir_path = os.path.join(data_dir_path, 'MITO_dataset')

ER_train_dir_path = os.path.join(ER_dataset_dir_path, 'train')
ER_val_dir_path = os.path.join(ER_dataset_dir_path, 'val')
ER_test_dir_path = os.path.join(ER_dataset_dir_path, 'test')

MITO_train_dir_path = os.path.join(MITO_dataset_dir_path, 'train')
MITO_val_dir_path = os.path.join(MITO_dataset_dir_path, 'val')
MITO_test_dir_path = os.path.join(MITO_dataset_dir_path, 'test')

ER_train_images_dir_path = os.path.join(ER_train_dir_path, 'images')
ER_train_masks_dir_path = os.path.join(ER_train_dir_path,'masks')
ER_val_images_dir_path = os.path.join(ER_val_dir_path, 'images')
ER_val_masks_dir_path = os.path.join(ER_val_dir_path,'masks')
ER_test_images_dir_path = os.path.join(ER_test_dir_path, 'images')
ER_test_masks_dir_path = os.path.join(ER_test_dir_path,'masks')

MITO_train_images_dir_path = os.path.join(MITO_train_dir_path, 'images')
MITO_train_masks_dir_path = os.path.join(MITO_train_dir_path,'masks')
MITO_val_images_dir_path = os.path.join(MITO_val_dir_path, 'images')
MITO_val_masks_dir_path = os.path.join(MITO_val_dir_path,'masks')
MITO_test_images_dir_path = os.path.join(MITO_test_dir_path, 'images')
MITO_test_masks_dir_path = os.path.join(MITO_test_dir_path,'masks')

ER_train_images_path_list = [os.path.join(ER_train_images_dir_path, x) for x in os.listdir(ER_train_images_dir_path)]
ER_train_masks_path_list = [os.path.join(ER_train_masks_dir_path, x) for x in os.listdir(ER_train_masks_dir_path)]
ER_val_images_path_list = [os.path.join(ER_val_images_dir_path, x) for x in os.listdir(ER_val_images_dir_path)]
ER_val_masks_path_list = [os.path.join(ER_val_masks_dir_path, x) for x in os.listdir(ER_val_masks_dir_path)]
ER_test_images_path_list = [os.path.join(ER_test_images_dir_path, x) for x in os.listdir(ER_test_images_dir_path)]
ER_test_masks_path_list = [os.path.join(ER_test_masks_dir_path, x) for x in os.listdir(ER_test_masks_dir_path)]

MITO_train_images_path_list = [os.path.join(MITO_train_images_dir_path, x) for x in os.listdir(MITO_train_images_dir_path)]
MITO_train_masks_path_list = [os.path.join(MITO_train_masks_dir_path, x) for x in os.listdir(MITO_train_masks_dir_path)]
MITO_val_images_path_list = [os.path.join(MITO_val_images_dir_path, x) for x in os.listdir(MITO_val_images_dir_path)]
MITO_val_masks_path_list = [os.path.join(MITO_val_masks_dir_path, x) for x in os.listdir(MITO_val_masks_dir_path)]
MITO_test_images_path_list = [os.path.join(MITO_test_images_dir_path, x) for x in os.listdir(MITO_test_images_dir_path)]
MITO_test_masks_path_list = [os.path.join(MITO_test_masks_dir_path, x) for x in os.listdir(MITO_test_masks_dir_path)]

def select_element_having_substr(substr: str, path_list: list[str]) -> list[str]:
    """
    Selects an element from the path list that contains the given substring.

    Args:
        substr: Substring to search for in the path list elements.
        path_list: List of paths to search within.

    Returns:
        The selected path containing the substring.
    """
    selected_list = [path for path in path_list if substr in path]
    assert len(selected_list) == 1, f'The {substr} is not contained or unique in the path list.'
    return selected_list[0]

def merge_images_masks_path_list(images_path_list: list[str], masks_path_list: list[str]) -> list[dict[str, dict[str, str]]]:
    """
    Merge images and masks path lists into a list of dictionaries.

    Args:
        images_path_list: List of paths to image files.
        masks_path_list: List of paths to mask files.

    Returns:
        List of dictionaries where each dictionary contains image and mask paths.
    """
    assert len(images_path_list) == len(masks_path_list), 'The number of images and masks should be the same.'
    merged_list = []
    for image_path in images_path_list:
        match = re.search(r'(.+?)\.tif', image_path)
        if match:
            name = os.path.basename(match.group(1))
            masks_path = select_element_having_substr(substr=name, path_list=masks_path_list)
            merged_list.append({name: {'image': image_path, 'mask': masks_path}})
        else:
            print(f'The image path {image_path} is not ending with ".tif"')
            exit(1)
    return merged_list

ER_train_path_list = merge_images_masks_path_list(ER_train_images_path_list, ER_train_masks_path_list)
ER_val_path_list = merge_images_masks_path_list(ER_val_images_path_list, ER_val_masks_path_list)
ER_test_path_list = merge_images_masks_path_list(ER_test_images_path_list, ER_test_masks_path_list)

MITO_train_path_list = merge_images_masks_path_list(MITO_train_images_path_list, MITO_train_masks_path_list)
MITO_val_path_list = merge_images_masks_path_list(MITO_val_images_path_list, MITO_val_masks_path_list)
MITO_test_path_list = merge_images_masks_path_list(MITO_test_images_path_list, MITO_test_masks_path_list)

# Output directory
output_dir_path = os.path.join(os.getcwd(), '.', 'output')
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

''' Hyperparameters '''
class Hyperparameters:
    batch_size = 64
    learning_rate = 1e-4

    @staticmethod
    def make_subdir(exp_id : int, dataset: str, loss: str, optimizer: str, epochs : int) -> str:
        if exp_id == 1:
            exp_name = 'exp1_loss_opt'
        elif exp_id == 2:
            exp_name = 'exp2_epochs'
        elif exp_id == 3:
            exp_name = 'exp3_SGD'
        else:
            raise NotImplementedError(f'Invalid experiment ID={exp_id}.')
        
        subdir_name = ''.join([dataset, '_', loss, '_', optimizer, '_', str(epochs)])
        subdir_name = os.path.join(exp_name, subdir_name)
        subdir_path = os.path.join(output_dir_path, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        return subdir_path