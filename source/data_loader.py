import pandas as pd
import numpy as np
from os.path import join
from scipy.io import arff


# Large 2000+
def load_data(filename):
    data_path = '../data'
    if filename.endswith('.arff'):
        data = arff.loadarff(join(data_path, filename))
        df = pd.DataFrame(data[0])
    else:
        df = pd.read_csv(join(data_path, filename), sep=',|;', engine='python')

    y = df.to_numpy()[:, -1].astype(str)
    X = df.to_numpy()[:, :].astype(str)
    x_names = {}
    for i, v in enumerate(list(df.columns)[:-1]):
        x_names[v + '__' + str(i)] = np.unique(X[:, i])
    return X, y, x_names, np.unique(X[:, -1])


# Medium 500-2000

# Small 0-500
if __name__ == "__main__":
    X, y, x_names, y_names = load_data('mushroom.csv')
    X, y, x_names, y_names = load_data('divorce.csv')
    X, y, x_names, y_names = load_data('student_performance.arff')
    print(1)
