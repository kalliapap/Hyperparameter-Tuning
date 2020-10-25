import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from header import report, writeResultsToFile, datasetDESCRtoFile, modelEvaluation
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
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=45)

filename = "results/knn/knn_output.txt"
#datasetDESCRtoFile(filename, dataset)

knn = KNeighborsClassifier(algorithm='kd_tree')
base_model = knn.fit(X_train, y_train)

'''
=================================================
                   GridSearchCV
=================================================
'''

leaf_size = list(range(1,10))
n_neighbors = list(range(1,6))
p=[1,2]
param_grid = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size, 'p': p}

grid_search = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), param_grid=param_grid, cv=10, n_jobs=-1)
start = time()
grid_search.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(grid_search.cv_results_['params'])))

f = open(filename, "w")
f.write('KNN Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%grid_search.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

best_results = report(grid_search.cv_results_)

print("\n KNN Grid Search Best Estimator on Training Set: ", grid_search.best_estimator_,"\n")
print("KNN GridSearchCV Best Parameters: ", grid_search.best_params_)

print("=========Evaluation Results on Test Set========\n")
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = grid_search.predict(X_test)
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
plt.title('KNN GridSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/knn/knn_grid.png')
print(classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
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
plt.title('KNN GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/knn/knn_roc.png')
print("===================================================================================")
'''
==================================================
               RandomizedSearchCV
==================================================
'''

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
param_grid = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size, 'p': p}

# New subset for randomized search
X_train_rand, z, y_train_rand, w = train_test_split(X_train, y_train,
                                                    test_size=0.3, random_state=45)

knn_random = RandomizedSearchCV(KNeighborsClassifier(algorithm='kd_tree'), param_grid, cv=10, n_jobs=-1)
start = time()
knn_model = knn_random.fit(X_train_rand, y_train_rand)
end = time() - start
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(knn_random.cv_results_['params'])))

f = open(filename, "a")
f.write('KNN Random Search Results\n\n')
f.write("Best parameters set found on development subset:")
f.write('\n\n')
f.write('%s'%knn_random.best_params_)
f.write('\n\n')
f.write("Grid scores on development subset:")
f.write('\n\n')
means = knn_random.cv_results_['mean_test_score']
stds = knn_random.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, knn_random.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

random_bestResults = report(knn_random.cv_results_)
print("KNN RandomSearchCV Best Parameters: ", knn_random.best_params_)
print("KNN RandomSearchCV Best Estimator on Training Subset: ", knn_random.best_estimator_)

print("=========Evaluation Results on Test Set========\n")
y_pred = knn_random.predict(X_test)
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
plt.title('KNN RandomizedSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/knn/knn_random.png')
print(classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = knn_random.predict_proba(X_test)[:, 1]
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
plt.title('KNN RandomizedSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/knn/knn_random_roc.png')
print("===================================================================================")
'''
=================================================
               Final GridSearchCV
#=================================================
'''
leaf_size = []
n_neighbors = []
p = []
for i in range(0, len(random_bestResults)):
    leaf_size.append(random_bestResults[i]['leaf_size'])
    n_neighbors.append(random_bestResults[i]['n_neighbors'])
    p.append(random_bestResults[i]['p'])
leaf_size = list(set(leaf_size))
n_neighbors = list(set(n_neighbors))
p = list(set(p))

grid = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size, 'p': p}

grid_search_final = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), param_grid=grid, cv=10, n_jobs=-1)
start = time()
grid_search_final.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(grid_search_final.cv_results_['params'])))

f = open(filename, "a")
f.write('KNN Final Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%grid_search_final.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = grid_search_final.cv_results_['mean_test_score']
stds = grid_search_final.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_final.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

report(grid_search_final.cv_results_)
print("\n KNN Grid Search Best Estimator on Training Set: ", grid_search_final.best_estimator_,"\n")
print("KNN GridSearchCV Best Parameters: ", grid_search_final.best_params_)

print("=========Evaluation Results on Test Set========\n")
y_pred = grid_search_final.predict(X_test)
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
plt.title('KNN Final GridSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/knn/knn_Finalgrid.png')
print("Classification Report on Test Set",classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_pred_proba = grid_search_final.predict_proba(X_test)[:, 1]
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
plt.title('KNN Final GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/knn/knn_final_roc.png')
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
gridBest_accuracy = modelEvaluation(filename, "GridSearch Best Model", grid_search, grid_search.best_params_,  X_test, y_test)
randomBest_accuracy = modelEvaluation(filename, "RandomizedSearch Best Model", knn_random, knn_random.best_params_,  X_test, y_test)
gridSearch_accuracy = modelEvaluation(filename, "GridSearch Best Model", grid_search_final, grid_search_final.best_params_, X_test, y_test)
f = open(filename, "a")
f.write("=======================================================\n")
f.close()