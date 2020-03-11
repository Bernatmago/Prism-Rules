from data_loader import load_data
import numpy as np

# X, y, x_names, y_names = load_data('mushroom.csv')
# X, y, x_names, y_names = load_data('divorce.csv')
X, y, x_names, y_names = load_data('student_performance.arff')

for cls in y_names:
    # Restore dataset
    # X = np.copy(X_init)
    # y = np.copy(y_init)
    idx_train = np.arange(X.shape[0])
    # Repeat until all instances of class x are removed from training set
    cls_rules = []
    while np.count_nonzero(y[idx_train] == cls) > 0:
        idx_sub = np.copy(idx_train)

        rule = []
        # Repeat while subset contains only instances of the class
        while np.count_nonzero(y[idx_sub] != cls) > 0:
            max_prob = 0
            max_idx = []
            max_v = ''
            for name, values in x_names.items():
                att_name, att_idx = name.split('__')
                for v in values:
                    # Mirar tema indices y probs bugueao
                    v_idx = np.intersect1d(np.where(X[:, int(att_idx)] == v)[0], idx_sub)
                    # Pick best value and filter only containing the selected
                    if np.count_nonzero(y[v_idx] == cls) == 0:
                        prob = 0
                    else:
                        prob = np.count_nonzero(y[v_idx] == cls) / v_idx.shape[0]
                    if prob > max_prob:
                        # Get global indexes to filter training set
                        max_prob = prob
                        max_idx = np.copy(v_idx)
                        max_v = att_name + '=' + v
            # Treure elements usats
            rule.append(max_v)
            idx_sub = np.copy(max_idx)
        idx_train = np.setdiff1d(idx_train,idx_sub)
        cls_rules.append(' ^ '.join(rule))
        # Remove the instances from the training set
    print(cls, cls_rules)