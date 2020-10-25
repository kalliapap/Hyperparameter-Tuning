from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm, datasets, linear_model, ensemble, tree, neighbors, metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# load data
dataset = datasets.load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[dataset['data'], dataset['target']], columns = np.append(dataset['feature_names'], ['target']))
df_cancer.head()

#orange benign, blue maligant test

plt.figure(figsize = (20,20))
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
#plt.title('Breast Cancer Dataset Pairplot', fontsize = 20)
plt.savefig('results/pairplot.png')

#ax = plt.axes()
plt.figure(figsize = (20,20))
sns.heatmap(df_cancer.corr(), annot=True, fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap', fontsize = 20)
plt.savefig('results/corr_heatmap.png')

X, y = dataset.data, dataset.target
# relabel so the maligant class is positive
for i in range(len(y)):
    if y[i] > 0:
        y[i] = 0
    elif y[i] < 1:
        y[i] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=45)
print(y_train)
classifiers = [svm.SVC(probability=True), ensemble.RandomForestClassifier(),neighbors.KNeighborsClassifier()]
i=0
for classifier in classifiers:
    i = i+1
    start = time()
    model = classifier.fit(X_train, y_train)
    end = time() - start
    print("Fit took %.2f seconds for classifier %s with default parameters.\n"
          % (end, classifier))
    print(classifier)
    pred = model.predict(X_test)
    conf = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print('tn= ',tn,'\nfp= ', fp,'\nfn= ', fn, '\ntp= ',tp)
    print('Confusion Matrix :')
    print(conf)
    print('Accuracy Score: ', metrics.accuracy_score(y_test, pred))
    print('Precision Score: ', metrics.precision_score(y_test, pred))
    print('Recall Score: ', metrics.recall_score(y_test, pred))
    print('Report:')
    print(classification_report(y_test, pred))
    confusion = pd.DataFrame(conf, index=['Beinign','Maligant'],
                                 columns=['predicted_benign', 'predicted_maligant'])
    print(confusion)

    sns.set(font_scale=1.4)
    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion, annot=True, fmt='g')
    plt.savefig('results/baseline/heatmap_'+str(i)+'.png')

    # Use the probabilities for your ROC and Precision-recall curves
    y_score = model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    plt.figure(figsize=(20, 10))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label="auc = " + str(auc))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.title(str(i)+' ROC Curve', fontsize=20)
    plt.savefig('results/baseline/roc'+str(i)+'.png')

