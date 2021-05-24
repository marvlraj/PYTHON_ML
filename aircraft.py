from pandas.plotting import scatter_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
os.chdir("C:\\Users\\Sameer\\Downloads\\LEARN PYTHON(ML)\\ML REGRESSION")
aircarft = pd.read_csv("AIRCARFT.csv")
# print(aircarft.describe())
data = aircarft["Number of O-rings at risk on a given flight"].value_counts()
print(data)
data = (aircarft.hist(bins=50, figsize=(20, 15)))
# print(plt.show())
train_set , test_set = train_test_split(aircarft, test_size=0.2, random_state=42)
attributes = ["Number of O-rings at risk on a given flight", "Number experiencing thermal distress",
              "Launch temperature (degrees F)", "Leak-check pressure (psi)", "Temporal order of flight"]
scatter_matrix(aircarft[attributes], figsize=(12, 8))
# print(plt.show())

train_set, test_set = train_test_split(aircarft, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(aircarft, aircarft['Number of O-rings at risk on a given flight']):
    strat_train_set = aircarft.loc[train_index]
    strat_test_set = aircarft.loc[test_index]
airflow = strat_train_set.copy()

airflow = strat_train_set.drop("Number of O-rings at risk on a given flight", axis=1)
airflow_labels = strat_train_set["Number of O-rings at risk on a given flight"].copy()

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),
])
airflow_num_tr = my_pipeline.fit_transform(airflow)
imputer = SimpleImputer(strategy="median")
imputer.fit(airflow)
X = imputer.transform(airflow)
airflow_tr = pd.DataFrame(X, columns=airflow.columns)
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
model = LinearRegression()
model.fit(airflow_num_tr, airflow_labels)

some_data = airflow.iloc[:5]
some_labels = airflow_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
a = model.predict(prepared_data)
# print(a)
print(list(some_labels))
airflow_predictions = model.predict(airflow_num_tr)
mse = mean_squared_error(airflow_labels, airflow_predictions)
rmse = np.sqrt(mse)
# print(mse)
scores = cross_val_score(model, airflow_num_tr, airflow_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standerd deviation: ", scores.std())


print_scores(rmse_scores)
X_test = strat_train_set.drop("Temporal order of flight", axis=1)
Y_test = strat_train_set["Temporal order of flight"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
x =  final_rmse
int(x)
with open("air.txt", "w") as f:
    air = int(f.write(str(x)))

print(final_predictions,list(Y_test))