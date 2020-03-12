from data_loader import load_data
from prism import Prism
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

for dataset in ['mushroom.csv', 'divorce.csv', 'student_performance.arff']:
    print ('PRISM on ' + dataset)
    X, y, names = load_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    prism = Prism()
    prism.fit(X_train, y_train, names)
    preds = prism.predict(X_test)
    print(classification_report(y_test, preds))