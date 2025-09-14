import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = iris.target_names[y]
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size= 0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: ", accuracy)

#Example

New_Flower = [[5,4,3,1]]
prediction = model.predict(New_Flower)
print("Predicted Species: ", iris.target_names[prediction[0]])


