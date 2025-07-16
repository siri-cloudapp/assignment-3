import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.dropna()
X = df[['Pclass', 'Age', 'SibSp', 'Fare']]
y = df['Survived']

model = RandomForestClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
