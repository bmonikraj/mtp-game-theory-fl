import pandas as pd
import os
import glob

datasets = ['fmnist', 'cifar10', 'cifar100']

base_path = 'results'

for d in datasets:
    files = glob.glob(os.path.join(base_path, d, '*.csv'))
    print(f"Dataset = {d}")
    for f in files:
        df = pd.read_csv(f)
        _K, _h = f.split('_')[0], f.split('_')[1]
        _K = _K.split('=')[1]
        _h = _h.split('=')[1]
        if _K == '5':
            heterogenity = 'EXTREME'
        else:
            if _h == "True":
                heterogenity = 'HOMOGENOUS'
            else:
                heterogenity = 'SEVERE'
        print(f"Heterogenity = {heterogenity} | Average accuracy = {df['gt_fedavg_acc'].mean()}")
    print("")
