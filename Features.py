import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
from numpy import sqrt
from sklearn.metrics import mean_squared_error
import math

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

# Loading and Visualizing the Dataset:
series=pd.read_csv("shampoo_sales.csv", header=0, parse_dates=[0], index_col=0, date_parser=parser)
print(series.head())
series.plot()
# plt.show()

# The distribution using histogram
fig, ax = plt.subplots(figsize=(15, 6))
sns.displot(series["Sales"], kde = True, color = 'blue', bins = 30, ax = ax)
ax.set_title("Distribution of Sales")
# The Distribution of ‘Sales’  Using Box Plot:
series["Sales"].plot(kind="box", vert=False, title="Distribution of Sales",ax=ax)
# plt.show()

# .....................................................
# ...........The Result................................
# As seen from the two plots, we do not see any irregular distribution or outliers.
# It’s essential to see how the data move over time, so let’s create a time
# series plot.
#.............end Result...............................

# #convert to dataframe
# series= series["Sales"]. resample("1M").mean().fillna(method="ffill").to_frame()
# print(series)

# we need to take our target ‘Sales’  and manipulate it to turn it into a
# feature by creating a lag by shifting # our data. In other words, we will use
# the previous month’s sale to predict the present.

series["Sales.L1"] = series["Sales"].shift(2)
series.dropna(inplace = True)
print(series.head())
# Now we have a feature and a target, and we need to see if there
# is a relationship between these two things, the way to do this
# is the correlation.

print(series.corr())
# ...........The Result................................
# Sales     1.000000  0.719482
# Sales.L1  0.719482  1.000000
# As seen, there’s a strong correlation between what happened the
# previous month and what happens this month.
#.............end Result...............................
# Split Data...........................................
# spliting data into feature X and target Y
# Then spliting X and Y into train and test data
# Split the data into featuer and target
target = "Sales"
y = series[target]
X = series.drop(columns= target)
#Split the data into train and test sets
cutoff = int(len(X) * 0.8)
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]
# Building a linear regression model named model and fit it to our training data.
model = LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
df_pred_test = pd.DataFrame(
             {
             "y_test": y_test,
             "y_pred": model.predict(X_test)
             }
)
print(df_pred_test.head())

fig = px.line(df_pred_test, labels= {"value": "Sales"}, title = "Linear Regression Model: Actual Sales vs. Predicted Sales.")
fig.show()

