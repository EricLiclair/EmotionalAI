# imports
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# dataset
dataset = "data/emotions.csv"
data = pd.read_csv(dataset)

# label
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].replace(label_mapping)

# splitting
X = data.drop('label', axis=1).copy()
y = data['label'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=np.random)

# training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# export model
model_name = "data/model.sav"
with open(model_name, "wb") as model:
  pickle.dump(clf, model)

print(y_test == clf.predict(X_test))

