import pandas as pd
from os.path import join
from scipy.io import arff

# Large 2000+
def load_data(filename):
    data_path = './data'
    if filename.endswith('.arff'):
        data = arff.loadarff(join(data_path,filename))
        df = pd.DataFrame(data[0])
    else:    
        df = pd.read_csv(join(data_path,filename))

    y = df.to_numpy()[:,-1]
    X = df.to_numpy()[:,:-1]
    return X, y
# Medium 500-2000

# Small 0-500
if __name__ == "__main__":
    X, y = load_data('mushroom.csv')
    X, y = load_data('divorce.csv')
    X, y = load_data('student_performance.arff')
    pass