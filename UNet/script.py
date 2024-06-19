"""
Script: for different experiments
"""

import os
import shutil

# Experiment 1: Different combinations of Loss Function and Optimizer
exp_id = 1
for dataset in ['ER', 'MITO']:
    for loss_function in ['BCELoss', 'DICELoss', 'IoULoss']:
        for optimizer in ['Adam', 'SGD', 'AdamW', 'RMSProp']:
            for epochs in [300]:
                return_value = os.system(f'python main.py --exp_id {exp_id} --dataset {dataset} --loss_function {loss_function} --optimizer {optimizer} --epochs {epochs}')
                if return_value!= 0:
                    print(f'Error: Experiment 1 with {dataset}, {loss_function}, {optimizer}, {epochs} failed.')
                    exit(1)


# Experiment 2: Different epochs for different optimizers
exp_id = 2
for dataset in ['ER', 'MITO']: 
    loss_function = 'IoULoss'
    optimizer = 'RMSProp'
    for epochs in [100, 150, 200, 400, 500, 600]:
        return_value = os.system(f'python main.py --exp_id {exp_id} --dataset {dataset} --loss_function {loss_function} --optimizer {optimizer} --epochs {epochs}')
        if return_value!= 0:
            print(f'Error: Experiment 2 with {dataset}, {loss_function}, {optimizer}, {epochs} failed.')
            exit(1)

# Experiment 3: Different SGD parameters
exp_id = 3
for dataset in ['ER', 'MITO']:
    loss_function  = 'IoULoss'
    epochs = 300
    for optimizer in ['SGD_momentum', 'SGD_weightdecay', 'SGD_nesterov', 'SGD_all']:
        return_value = os.system(f'python main.py --exp_id {exp_id} --dataset {dataset} --loss_function {loss_function} --optimizer {optimizer} --epochs {epochs}')
        if return_value!= 0:
            print(f'Error: Experiment 2 with {dataset}, {loss_function}, {optimizer}, {epochs} failed.')
            exit(1)

pycache_path = os.path.join('.', '__pycache__')
if os.path.exists(pycache_path):
    shutil.rmtree(pycache_path)
print('\nDone!')
