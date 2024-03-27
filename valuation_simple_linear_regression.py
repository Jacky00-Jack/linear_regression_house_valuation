import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get data
df = pd.read_csv(r"C:\Users\New User\Downloads\Housing Price Analysis.csv\Housing Price Analysis.csv")
print(df.head().T)

#examine data types
print(df.info())

#check number of unique values for Ocean Proximity
print(df['ocean_proximity'].value_counts())

#make the headers neat by converting to lower case and subs spaces with '-'
df['ocean_proximity'] = df['ocean_proximity'].str.lower().replace('[^0-9a-zA-Z]+','_',regex=True)

#convert the categorical data into binaries using get_dummies
encodings = pd.get_dummies(df['ocean_proximity'], prefix='proximity')
df = pd.concat([df, encodings], axis=1)
print(df.sample(5).T)

#create histograms to visualise data and removes outliers
#bins is to determine the intervals within the range
df.hist(bins=50, figsize=(20,15))
plt.show()

#Check the summary statistic
print(df.describe().T)

#Check correlation between data
plt.figure(figsize=(14,8))
corr = df.corr(numeric_only=True)
heatmap = sns.heatmap(corr, annot=True, cmap="Blues")
plt.show()

#plot barchart to visualise the correlation of house value to variables
plt.figure(figsize=(14,8))
bars = df.corr(numeric_only=True)['median_house_value'].sort_values(ascending=False).plot(kind='bar')
plt.show()

#make a scatterplot to visualise the correlation of value against correlation
df.plot(
    x='longitude', y='latitude',
    kind='scatter', figsize=(10,7),
    alpha=.4,
    c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True
)
plt.show()

#check for missing value
print(df.isnull().sum())

#can either fill missing data with 0 or mean data
#since the amount of data missing is quite few, so filling with 0 does not impact much
df = df.fillna(0)

#build regression model but dropping ocean proximity since it is not a numerical data
#also dropping median house value since it is the predicted value, hence it is labelled as y
X = df.drop(['median_house_value','ocean_proximity'], axis=1)
y = df['median_house_value']

#regression model will then be trained using the data
#test size 0.3 is to set aside 30% of the data at random to test the model
#4 data sets will be x_train, y_train, x_test, y_test
#x_train to train the initial model, y_train to self check
#then the x_test is used to test the model as it will be a never seen data, and the y_test will use to validate it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#scaler is used to make sure the data fits into the model
#fit_transform(fit here means learn) is used on x_train data and transform(without learning) on x_test to ensure no data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#linear regression is used to learn the model
#fit is called to learn the x_train and y_train
model = LinearRegression()
model.fit(X_train, y_train)

#x_test is now used within the model to carry out prediction and data is stored as y_pred
y_pred = model.predict(X_test)

#y_pred will be an array, root mean squared error will be used to assess the error of this model
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred))) 

#error here can be compared to the actual mean to determine the performance
#a graph can be plotted using the y_pred and y_test to visualise the error
test = pd.DataFrame({'Predicted value':y_pred, 'Actual value':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual value','Predicted value'])
plt.show()