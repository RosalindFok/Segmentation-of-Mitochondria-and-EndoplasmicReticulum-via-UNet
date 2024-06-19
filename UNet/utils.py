"""
utils.py: contains utility functions for the project
"""

import os
import cv2
import csv
import json
import numpy as np

__all__ = ['read_json', 'write_json', 'write_csv_from_dict', 'write_tif']

def write_json(data : dict[str, dict[str, float]], output_file : str) -> bool:
    """
    Write the data to a JSON file.

    Args:
        data (dict[str, dict[str, float]]): The data to write to the JSON file.
        output_file (str): The path to the output JSON file.

    Returns:
        bool: True if the JSON file was successfully saved, False otherwise.
    """
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON file saved to {output_file}")
    return True

def write_csv_from_dict(data : dict[str, dict[str, float]], output_file : str) -> bool:
    """
    Write the data to a CSV file from the dictionary.

    Args:
        data (dict[str, dict[str, float]]): The data in dictionary format to write to the CSV file.
        output_file (str): The path to the output CSV file.

    Returns:
        bool: True if the CSV file was successfully saved, False otherwise.
    """
    matrix = []
    for key, value in data.items():
        for k, v in value.items():
            matrix.append([key, k, v])
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Metric', 'Score'])
        for row in matrix:
            writer.writerow(row)
    print(f"CSV file saved to {output_file}")
    return True

def write_tif(matrix : np.ndarray, output_dir : str, name : str) -> bool:
    """
    Write the matrix data to a TIFF file.

    Args:
        matrix (np.ndarray): The matrix data to write to the TIFF file.
        output_dir (str): The directory path to save the TIFF file.
        name (str): The name of the TIFF file.

    Returns:
        bool: True if the TIFF file was successfully saved, False otherwise.
    """
    output_dir = os.path.join(output_dir, f'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{name}.tif') if not name.endswith('.tif') else os.path.join(output_dir, name)

    # Check if the matrix values are within [0, 1] for probability matrix
    if np.min(matrix) >=0 and np.max(matrix) <= 1:
        matrix = (255 * matrix).astype(np.uint8)    
        cv2.imwrite(filename=output_file, img=matrix)
        return True
    else:
        raise NotImplementedError(f'Only probability matrices are supported for now.')