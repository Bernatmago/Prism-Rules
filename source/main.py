from data_loader import load_data
from prism import Prism
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    for (dataset, cls_idx) in [('mushroom.csv', -1), ('divorce.csv', -1), ('tic_tac_toe.csv', -1)]:
        print('PRISM on ' + dataset)
        X, y, names = load_data(dataset, cls_idx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
        prism = Prism()
        prism.fit(X_train, y_train, names)
        preds = prism.predict(X_test, y_test)
        print(prism)
        print(classification_report(y_test, preds))
