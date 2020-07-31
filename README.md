# PRISM ALGORITHM
Prism algorithm implementation done for the MAI-UPC Master

Datasets used:
* Mushroom: https://datahub.io/machine-learning/mushroom
* Tic-Tac-Toe Endgame: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
* Divorce: https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set

Thhe algorithm is contained into the Prism class.

Example:
```python
from prism import Prism
prism = Prism()
prism.fit(X_train, y_train, names)
preds = prism.predict(X_test, y_test)
```
