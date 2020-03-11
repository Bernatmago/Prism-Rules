from data_loader import load_data
import numpy as np


class Prism:
    def __init__(self):
        self.X = np.zeros(0)
        self.y = np.zeros(0)
        self.X_names = {}
        self.y_names = []
        pass

    def __get_names(self, names):
        for i, v in enumerate(names[:-1]):
            self.X_names[v + '__' + str(i)] = np.unique(self.X[:, i])
            self.y_names = np.unique(X[:, -1])

    def __best_subset(self, cls, idx_sub):
        max_prob = 0
        max_idx = None
        max_v = None
        for name, values in self.X_names.items():
            att_name, att_idx = name.split('__')
            for v in values:
                v_idx = np.intersect1d(np.where(self.X[:, int(att_idx)] == v)[0], idx_sub)
                # Pick best value and filter only containing the selected
                if np.count_nonzero(self.y[v_idx] == cls) == 0:
                    prob = 0
                else:
                    prob = np.count_nonzero(y[v_idx] == cls) / v_idx.shape[0]
                if prob > max_prob or (prob == max_prob and len(v_idx) > len(v_idx)):
                    max_prob = prob
                    max_idx = np.copy(v_idx)
                    max_v = att_name + '=' + v
        return max_idx, max_v

    def __new_rule(self, cls, idx_train):
        idx_sub = np.copy(idx_train)
        rule = []
        while np.count_nonzero(self.y[idx_sub] != cls) > 0:
            idx_sub, best_attr = self.__best_subset(cls, idx_sub)
            rule.append(best_attr)
        idx_train = np.setdiff1d(idx_train, idx_sub)
        return idx_train, ' ^  '.join(rule)

    def __class_rules(self, cls):
        idx_train = np.arange(self.X.shape[0])
        cls_rules = []
        # Repeat until all instances of class x are removed from training set
        while np.count_nonzero(self.y[idx_train] == cls) > 0:
            idx_train, rule = self.__new_rule(cls, idx_train)
            cls_rules.append(rule)
        return cls_rules

    def fit(self, X, y, names):
        self.X = X
        self.y = y
        self.__get_names(names)

        for cls in self.y_names:
            rules = self.__class_rules(cls)
            print(rules)


if __name__ == '__main__':
    X, y, names = load_data('mushroom.csv')
    p = Prism()
    p.fit(X, y, names)
