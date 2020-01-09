{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#load libraries\n",
    "import pandas\n",
    "from pandas.plotting import scatter_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import model_selection\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "names = ['sepal length','sepal width','petal length','petal width','class']\n",
    "dataset = pandas.read_csv(url,names=names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(dataset.shape) #print rows and cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(dataset.head(10)) #print first 10 rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(dataset.describe()) #prints values or ranges of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(dataset.groupby('class').size()) #prints the class object attributes and size of it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.plot(kind = 'box', subplots = True, layout=(2,2),sharex=False,sharey=False) \n",
    "#plt.title('MY IRIS PLOT')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.hist()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scatter_matrix(dataset)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "array = dataset.values\n",
    "X = array[:,0:4]\n",
    "Y = array[:,4]\n",
    "validation_size = 0.20\n",
    "seed = 6\n",
    "X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "seed =6\n",
    "scoring = 'accuracy'\n",
    "\n",
    "#spot check algorithms\n",
    "\n",
    "models= []\n",
    "models.append(('LR', LogisticRegression()))\n",
    "models.append(('CART', DecisionTreeClassifier()))\n",
    "models.append(('KNN', KNeighborsClassifier()))\n",
    "models.append(('LDA', LinearDiscriminantAnalysis()))\n",
    "#models.append(('NB', GaussianNB))\n",
    "#models.append(('SVM', SVC))\n",
    "\n",
    "#evaluate each model in turn\n",
    "\n",
    "results= []\n",
    "names = []\n",
    "for name,model in models:\n",
    "    kfold = model_selection.KFold(n_splits=10,random_state =seed)\n",
    "    cv_results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    msg = \"%s: %f (%f)\" % (name, cv_results.mean(), cv_results.std())\n",
    "    print(msg)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
