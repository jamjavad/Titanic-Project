Titanic Survival Prediction

Machine Learning project to predict passenger survival on the Titanic

```
python
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv',index_col=0)
test  = pd.read_csv('input/test.csv')
```

```
python

train.info()

test.info()

```