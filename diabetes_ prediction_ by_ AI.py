import pandas as pd

v=pd.read_csv(r"H:\STUDY FILES\THESIS\ml works\diabetes2.csv")

print(v.shape)

x=v.iloc[:,:8]

print(x)

y=v.iloc[:,8]

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.15)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_pred=dtc.predict(x_test)

print(y_test[0:10])

print(y_pred[0:10])

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

a=confusion_matrix(y_test,y_pred)

#defining accuracy
print((a[0,0]+a[1,1])/116)













