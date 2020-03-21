import pandas as pd
import numpy as np
from os.path import join
from scipy.io import arff


def load_data(filename, pred_idx=-1):
    data_path = '../data'
    if filename.endswith('.arff'):
        data = arff.loadarff(join(data_path, filename))
        df = pd.DataFrame(data[0])
    else:
        df = pd.read_csv(join(data_path, filename), sep=';', engine='python')
    X = np.delete(df.to_numpy().astype(str), pred_idx, axis=1)
    y = df.to_numpy()[:, pred_idx].astype(str)
    names = list(df.columns)
    names.pop(pred_idx)
    return X, y, names

if __name__ == "__main__":
    X, y, names = load_data('mushroom.csv', -1)
    X, y, names = load_data('divorce.csv', -1)
    print(1)
