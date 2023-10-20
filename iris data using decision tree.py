import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data1=pd.read_csv("Iris.csv")
data=data1.iloc[:,1:]
data.head()
x=data.iloc[:,:4]
y=data.iloc[:,4]
data.describe()
# import seaborn as sns
# sns.pairplot(data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier()
model_dtc.fit(x_train,y_train)
y_predict=model_dtc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))