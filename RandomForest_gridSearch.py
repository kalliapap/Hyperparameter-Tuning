from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from header import report, bestResults, writeResultsToFile, modelEvaluation, datasetDESCRtoFile
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# load data
dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target
# relabel so the maligant class is positive
for i in range(len(y)):
    if y[i] > 0:
        y[i] = 0
    elif y[i] < 1:
        y[i] = 1
print('Data:', X.shape)
print('Target:', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=45)

# write results to file
filename = "results/rf/output_rf.txt"
#datasetDESCRtoFile(filename, dataset)

#build base model classifier
rf = RandomForestClassifier()
base_model = rf.fit(X_train, y_train)

'''
=================================================
                   GridSearchCV
=================================================
'''
# hyperparameter tuning values
n_estimators = [200, 600, 1200, 1800, 2000]
max_depth = [20, 50, 80, 100, None]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]

# Create the grid
grid = {'n_estimators': n_estimators,
         'max_depth': max_depth,
         'min_samples_split': min_samples_split,
         'min_samples_leaf': min_samples_leaf}

rf_search = GridSearchCV(RandomForestClassifier(), grid, cv=10, n_jobs=-1)
start = time()
rf_search.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(rf_search.cv_results_['params'])))

f = open(filename, "w")
f.write('Random Forest Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%rf_search.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = rf_search.cv_results_['mean_test_score']
stds = rf_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rf_search.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

best_results = report(rf_search.cv_results_)
print("RF GridSearchCV Best Parameters: ", rf_search.best_params_)
print("\n GridSearchCV Best Estimator on Training set: ", rf_search.best_estimator_,"\n")

print("=========Evaluation Results on Test Set========\n")
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = rf_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix, classification_report
cm = np.array(confusion_matrix(y_test, y_pred))
confusion = pd.DataFrame(cm, index=['Benign', 'Maligant'],
                         columns=['predicted_benign', 'predicted_maligant'])
confusion

f = open(filename, "a")
f.write("Detailed classification report:\n\n")
f.write("The model is trained on the full development set.\n\n")
f.write("The scores are computed on the full evaluation set.\n\n")
f.write('%s\n'%classification_report(y_test, y_pred))
f.write('Accuracy Score: %s\n' % accuracy_score(y_test, y_pred))
f.write('Precision Score: %s\n'% precision_score(y_test, y_pred))
f.write('Recall Score: %s\n'% recall_score(y_test, y_pred))
f.write('============================================\n\n')
f.close()

sns.set(font_scale=1.4)
plt.figure(figsize = (12,12))
sns.heatmap(confusion, annot=True, fmt='g')
plt.title('RF GridSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/rf/rf_grid.png')
print(classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = rf_search.predict_proba(X_test)[:, 1]
# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('RF GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/rf/rf_roc.png')
print("===================================================================================")
'''
==================================================
               RandomizedSearchCV
==================================================
'''

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf
              }

X_train_rand, z, y_train_rand, w = train_test_split(X_train, y_train,
                                                    test_size=0.3, random_state=45)

rf_random = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=10, n_jobs=-1)
start = time()
rf_random.fit(X_train_rand, y_train_rand)
end = time() - start
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(rf_random.cv_results_['params'])))

f = open(filename, "a")
f.write('Random Forest Random Search Results\n\n')
f.write("Best parameters set found on development subset:")
f.write('\n\n')
f.write('%s'%rf_random.best_params_)
f.write('\n\n')
f.write("Grid scores on development subset:")
f.write('\n\n')
means = rf_random.cv_results_['mean_test_score']
stds = rf_random.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rf_random.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

random_bestResults = report(rf_random.cv_results_)
print("Random Forest RandomSearchCV Best Parameters: ", rf_random.best_params_)
print("Random Forest RandomSearchCV Best Estimator on Training Subset: ", rf_random.best_estimator_)

print("=========Evaluation Results on Test Set========\n")
y_pred = rf_random.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cm2 = np.array(confusion_matrix(y_test, y_pred))
confusion2 = pd.DataFrame(cm2, index=['Benign', 'Maligant'],
                         columns=['predicted_benign', 'predicted_maligant'])
confusion2

f = open(filename, "a")
f.write("Detailed classification report:\n\n")
f.write("The model is trained on the full development subset.\n\n")
f.write("The scores are computed on the full evaluation set.\n\n")
f.write('%s\n'%classification_report(y_test, y_pred))
f.write('Accuracy Score: %s\n' % accuracy_score(y_test, y_pred))
f.write('Precision Score: %s\n'% precision_score(y_test, y_pred))
f.write('Recall Score: %s\n'% recall_score(y_test, y_pred))
f.write('============================================\n\n')
f.close()

plt.figure(figsize = (12,12))
sns.heatmap(confusion2, annot=True, fmt='g')
plt.title('RF RandomizedSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/rf/rf_random.png')
print(classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = rf_random.predict_proba(X_test)[:, 1]
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('RF RandomizedSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/rf/rf_random_roc.png')
print("===================================================================================")
'''
=================================================
              Final GridSearchCV
#=================================================
'''
# gia tis kaluteres times tou randomizedSearch kanw
# ksana gridSearch me perissoterous sundasmous parametrwn
# kai exhaustive anazhthsh
n_estimators = []
max_features = []
max_depth = []
min_samples_split = []
min_samples_leaf = []
bootstrap = []

for i in range(0, len(random_bestResults)):
    n_estimators.append(random_bestResults[i]['n_estimators'])
    max_depth.append(random_bestResults[i]['max_depth'])
   #max_features.append(random_bestResults[i]['max_features'])
    min_samples_split.append(random_bestResults[i]['min_samples_split'])
    min_samples_leaf.append(random_bestResults[i]['min_samples_leaf'])
    #bootstrap.append(random_bestResults[i]['bootstrap'])

# removing duplicates values
n_estimators = list(set(n_estimators))
max_depth = list(set(max_depth))
#max_features = list(set(max_features))
min_samples_split = list(set(min_samples_split))
min_samples_leaf = list(set(min_samples_leaf))
#bootstrap = list(set(bootstrap))

grid = {'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
        }

gridSearch_model = GridSearchCV(RandomForestClassifier(), grid, cv=10, n_jobs=-1)
start = time()
gridSearch_model.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(gridSearch_model.cv_results_['params'])))

f = open(filename, "a")
f.write('Random Forest Final Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%gridSearch_model.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = gridSearch_model.cv_results_['mean_test_score']
stds = gridSearch_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gridSearch_model.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

report(gridSearch_model.cv_results_)
print("\n Grid Search Best Estimator on Training Set: ", gridSearch_model.best_estimator_,"\n")

best_model = gridSearch_model.best_estimator_

print("=========Evaluation Results on Test Set========\n")
y_pred = gridSearch_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cm3 = np.array(confusion_matrix(y_test, y_pred))
confusion3 = pd.DataFrame(cm3, index=['Benign', 'Maligant'],
                         columns=['predicted_benign', 'predicted_maligant'])
confusion3

f = open(filename, "a")
f.write("Detailed classification report:\n\n")
f.write("The model is trained on the full development set.\n\n")
f.write("The scores are computed on the full evaluation set.\n\n")
f.write('%s\n'%classification_report(y_test, y_pred))
f.write('Accuracy Score: %s\n' % accuracy_score(y_test, y_pred))
f.write('Precision Score: %s\n'% precision_score(y_test, y_pred))
f.write('Recall Score: %s\n'% recall_score(y_test, y_pred))
f.write('============================================\n\n')
f.close()

plt.figure(figsize = (12,12))
sns.heatmap(confusion3, annot=True, fmt='g')
plt.title('RF Final GridSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/rf/rf_final.png')
print(classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = gridSearch_model.predict_proba(X_test)[:, 1]
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('RF Final GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/rf/rf_final_roc.png')
print("===================================================================================")
'''
=================================================
              Accuracy Results
 Write accuracy score to file
 (modelEvaluation return accuracy score and write
 results to file)
=================================================
'''

base_accuracy = modelEvaluation(filename, "Base Model", base_model, None, X_test, y_test)
gridBest_accuracy = modelEvaluation(filename, "GridSearch Best Model", rf_search, rf_search.best_params_,  X_test, y_test)
randomBest_accuracy = modelEvaluation(filename, "RandomizedSearch Best Model", rf_random, rf_random.best_params_,  X_test, y_test)
gridSearch_accuracy = modelEvaluation(filename, "GridSearch Best Model", gridSearch_model, gridSearch_model.best_params_, X_test, y_test)
f = open(filename, "a")
f.write("=======================================================\n")
f.close()