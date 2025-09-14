import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = {
    'sqr_ft' : [900, 700, 800, 350, 241],
    'bedrooms' : [2, 3, 5, 7, 3],
    'price' : [10000, 20000, 50000, 100000, 20000]
}
df = pd.DataFrame(data)
print(df)

x = df[['sqr_ft', 'bedrooms']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Predicted Prices:', y_pred)
print('Actual Price: ', list(y_test))

mse = mean_squared_error(y_test, y_pred)
print('Error=',mse)

new_house = [[1500,4]]
predicted_price = model.predict(new_house)
print("Predicted Price for new house is:", predicted_price[0])
