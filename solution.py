import numpy as np
import pandas as pd
from sklearn import tree

#train_url="C:\Users\Samsung\Desktop\kickstart\train.csv"
train = pd.read_csv(r"C:\Users\Samsung\Desktop\kickstart\train.csv")

#test_url="C:\Users\Samsung\Desktop\kickstart\test.csv"
test = pd.read_csv(r"C:\Users\Samsung\Desktop\kickstart\test.csv")

#print(train.head())
#print(test.head())

#print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

train["Child"] = float('NaN')

train["Child"][train["Age"]>=18]=0
train["Child"][train["Age"]<18]=1

#print(train["Child"])

#test = test
test["Survived"] = 0
test["Survived"][test["Sex"] == 'female'] = 1
test["Survived"][test["Sex"] == 'male'] = 0

train["Sex"][train["Sex"] == 'male'] = 1
train["Sex"][train["Sex"] == 'female'] = 0

train["Emabarked"] = train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == 'S'] = 0
train["Embarked"][train["Embarked"] == 'Q'] = 1
train["Embarked"][train["Embarked"] == 'C'] = 2
train["Pclass"] = train["Pclass"].fillna(train["Pclass"].mean())
train["Sex"] = train["Sex"].fillna(train["Sex"].mean())
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Fare"] = train["Fare"].fillna(train["Fare"].mean())

target = train["Survived"].values
features = train[["Pclass","Sex", "Age"]].values
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(features, target)



test["Sex"][test["Sex"] == 'male'] = 1
test["Sex"][test["Sex"] == 'female'] = 0
test["Emabarked"] = test["Embarked"].fillna("S")

test["Embarked"][test["Embarked"] == 'S'] = 0
test["Embarked"][test["Embarked"] == 'Q'] = 1
test["Embarked"][test["Embarked"] == 'C'] = 2


test["Pclass"] = test["Pclass"].fillna(test["Pclass"].mean())
test["Sex"] = test["Sex"].fillna(test["Sex"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())

test["Child"] = float('NaN')

test["Child"][test["Age"]>=18]=0
test["Child"][test["Age"]<18]=1

t_features = test[["Pclass","Sex", "Age"]].values

predict = dtree.predict(t_features)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(predict, PassengerId, columns = ["Survived"])

my_solution.to_csv("C://Users//Samsung//Desktop//kickstart//my_solution.csv", index_label = ["PassengerId"])