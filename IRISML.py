
#load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal length','sepal width','petal length','petal width','class']
dataset = pandas.read_csv(url,names=names)

print(dataset.shape) #print rows and cols
print(dataset.head(10)) #print first 10 rows
print(dataset.describe()) #prints values or ranges of the dataset
print(dataset.groupby('class').size()) #prints the class object attributes and size of it

#plots
dataset.plot(kind = 'box', subplots = True, layout=(2,2),sharex=False,sharey=False)
plt.title('MY IRIS PLOT')
plt.show()
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)


seed =6
scoring = 'accuracy'

#spot check algorithms

models= []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model in turn

results= []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state =seed)
    cv_results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
