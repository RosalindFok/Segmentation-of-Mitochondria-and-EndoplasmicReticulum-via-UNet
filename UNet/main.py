"""
Project: Segmentation of Mitochondria and Endoplasmic Reticulum using U-Net
Author: <Rosalind Fok> <Yuechen Tao> <Yuxia Chai>
For: Biological Image Processing and Informatics, UCAS 2024 Spring
"""

import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import UNet
from config import Hyperparameters
from losses import BCELoss, DICELoss, IoULoss
from utils import write_json, write_csv_from_dict, write_tif
from metric import DICE, IoU, ACCURACY, SPECIFICITY, SENSITIVITY, HausdorffDistance
from dataset import ER_train_dataset, ER_val_dataset, ER_test_dataset
from dataset import MITO_train_dataset, MITO_val_dataset, MITO_test_dataset

def setup_device() -> torch.device:
    """
    Set up and return the available torch device.

    Returns:
        torch.device: A torch.device object representing the device,
        choosing GPU or CPU based on the availability of CUDA.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}')
    torch.cuda.init() if torch.cuda.is_available() else None
    return device

def train(device : torch.device,
          model : torch.nn.Module,
          loss_fn : torch.nn.Module,
          optimizer : torch.optim.Optimizer,
          train_dataloader : DataLoader
         ) -> tuple[float, torch.nn.Module]:
    """
    Train the model using the given parameters.

    Args:
        device (torch.device): The device on which to perform training.
        model (torch.nn.Module): The neural network model to train.
        loss_fn (torch.nn.Module): The loss function used for optimization.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_dataloader (DataLoader): The dataloader providing training data.

    Returns:
        tuple[float, torch.nn.Module]: A tuple containing the average training loss and the updated model.
    """
    model.train()
    train_loss = []
    torch.set_grad_enabled(True)
    for _, image, mask in train_dataloader:
        image = image.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        segmentation = model(image)
        loss = loss_fn(input=segmentation, target=mask)
        train_loss.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(train_loss) / len(train_loss), model

def valid(device : torch.device,
          model : torch.nn.Module,
          loss_fn : torch.nn.Module,
          valid_dataloader : DataLoader
         ) -> float:
    """
    Validate the model using the given parameters.

    Args:
        device (torch.device): The device on which to perform validation.
        model (torch.nn.Module): The neural network model to validate.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        valid_dataloader (DataLoader): The dataloader providing validation data.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for _, image, mask in valid_dataloader:
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            segmentation = model(image)
            loss = loss_fn(input=segmentation, target=mask)
            valid_loss.append(loss.item())
    return sum(valid_loss) / len(valid_loss)

def test(device : torch.device,
         model : torch.nn.Module,
         test_dataloader : DataLoader,
         metrics : list[torch.nn.Module],
         subdir_path : str
        ) -> dict[str, dict[str, float]]:
    """
    Perform testing of the model using the given parameters and metrics.

    Args:
        device (torch.device): The device on which to perform testing.
        model (torch.nn.Module): The neural network model to test.
        test_dataloader (DataLoader): The dataloader providing test data.
        metrics (list[torch.nn.Module]): List of evaluation metrics to calculate.
        subdir_path (str): The subdirectory path to save output.

    Returns:
        dict[str, dict[str, float]]: A dictionary containing evaluation metrics for each image and their averages.
    """
    model.eval()
    metrics_dict, average_metrics_dict = {}, {}
    with torch.no_grad():
        for name, image, mask in test_dataloader:
            name = name[0]
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            segmentation = model(image)
            # Save the segmentation result as tif file
            save_label = segmentation.cpu().numpy().squeeze()
            write_tif(matrix=save_label, output_dir=subdir_path, name=name)
            # metrics_dict = {{name: {metric_name: metric_value}, ...}， {...}}
            metrics_dict[name] = {}
            for key, metric in metrics.items():
                metric = metric(output=segmentation, target=mask)
                value = metric.item() if isinstance(metric, torch.Tensor) else metric
                metrics_dict[name][key] = value
                average_metrics_dict[key] = average_metrics_dict.get(key, 0) + value
    # Calculate the average of metrics for each image
    for key, value in average_metrics_dict.items():
        average_metrics_dict[key] = value / len(metrics_dict)
    metrics_dict['average'] = average_metrics_dict
    return metrics_dict

def parse_args() -> argparse.Namespace:
    # Select dataset, loss function, optimizer, epochs, exp_id
    parser = argparse.ArgumentParser(description='Select dataset, loss function, optimizer.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--loss_function', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--exp_id', type=int)
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()

    # Device
    device = setup_device()

    # Hyperparameters
    batch_size = Hyperparameters.batch_size
    learning_rate = Hyperparameters.learning_rate
    epochs = args.epochs
    set_dataset = args.dataset
    set_loss = args.loss_function
    set_optimizer = args.optimizer
    exp_id = args.exp_id
    subdir_path = Hyperparameters.make_subdir(
                                        exp_id=exp_id,      
                                        dataset=set_dataset, 
                                        loss=set_loss, 
                                        optimizer=set_optimizer, 
                                        epochs=epochs
                                    )

    # Data (ER and MITO): Set the batch_size of test_dataloader to 1 to get the metrics of each image.
    if set_dataset == 'ER':
        print(f'Dataset: Endoplasmic Reticulum')
        train_dataloader   = DataLoader(dataset=ER_train_dataset  , batch_size=batch_size, shuffle=False) 
        valid_dataloader   = DataLoader(dataset=ER_val_dataset    , batch_size=batch_size, shuffle=False) 
        test_dataloader    = DataLoader(dataset=ER_test_dataset   , batch_size=1         , shuffle=False) 
    elif set_dataset == 'MITO':
        print(f'Dataset: Mitochondria')
        train_dataloader = DataLoader(dataset=MITO_train_dataset, batch_size=batch_size, shuffle=False) 
        valid_dataloader = DataLoader(dataset=MITO_val_dataset  , batch_size=batch_size, shuffle=False) 
        test_dataloader  = DataLoader(dataset=MITO_test_dataset , batch_size=1         , shuffle=False) 
    else:
        raise ValueError('Invalid dataset name.')

    # Network
    model = UNet(n_channels=1, n_classes=1)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'The number of trainable parametes is {trainable_parameters}.')
    model = model.to(device=device)

    # Loss 
    if set_loss == 'BCELoss':
        print(f'Loss function: Binary Cross Entropy Loss')
        loss_fn = BCELoss()
    elif set_loss == 'DICELoss':
        print(f'Loss function: DICE Loss')
        loss_fn = DICELoss()
    elif set_loss == 'IoULoss':
        print(f'Loss function: Intersection over Union Loss')
        loss_fn = IoULoss()
    else:
        raise ValueError('Invalid loss function name.')

    # Metrics
    metrics = {
        'DICE' : DICE(), 
        'IoU' : IoU(), 
        'ACC' : ACCURACY(), 
        'SPE' : SPECIFICITY(), 
        'SEN' : SENSITIVITY(), 
        'HD' : HausdorffDistance()
    }

    # Optimizer
    print(f'Optimizer: {set_optimizer}')
    if set_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    elif set_optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif set_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0, weight_decay=0, nesterov=False)
    elif set_optimizer == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, 
                                        alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    elif set_optimizer == 'SGD_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0.9, weight_decay=0, nesterov=False)
    elif set_optimizer == 'SGD_weightdecay':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0, weight_decay=1e-5, nesterov=False)
    elif set_optimizer == 'SGD_nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0.9, weight_decay=0, nesterov=True) # nesterove is True -> momentum cannot be 0
    elif set_optimizer == 'SGD_all':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        raise ValueError('Invalid optimizer name.')

    # Train and Valid
    train_loss_dict, valid_loss_dict = {}, {}
    for ep in tqdm(range(epochs), desc='Epochs', leave=True):
        # Change the learning rate with epochs
        lr = learning_rate*((1-ep/epochs)**0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Train
        train_loss, model = train(
                                device=device, 
                                model=model, 
                                loss_fn=loss_fn, 
                                optimizer=optimizer, 
                                train_dataloader=train_dataloader
                            )
        train_loss_dict[ep] = train_loss
        # Valid
        valid_loss = valid(
                            device=device, 
                            model=model, 
                            loss_fn=loss_fn, 
                            valid_dataloader=valid_dataloader
                        )
        valid_loss_dict[ep] = valid_loss
    # Train/Valid Loss change with epochs
    write_json(data={'train_loss': train_loss_dict, 'valid_loss': valid_loss_dict}, 
               output_file=os.path.join(subdir_path, 'train_valid_loss.json'))

    # Test
    metrics_dict = test(
                        device=device, 
                        model=model, 
                        test_dataloader=test_dataloader, 
                        metrics=metrics, 
                        subdir_path=subdir_path
                    )
    # Metrics in Test save as json and csv
    write_json(data=metrics_dict, output_file=os.path.join(subdir_path, 'test_metrics.json'))
    write_csv_from_dict(data=metrics_dict, output_file=os.path.join(subdir_path, 'log.csv'))

if __name__ == '__main__':
    main()