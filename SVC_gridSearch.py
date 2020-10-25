import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import svm, datasets
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
filename = "results/svc/svc_output.txt"
# # datasetDESCRtoFile(filename, dataset)
#
# # build a base model classifier
clf = svm.SVC()
base_model = clf.fit(X_train, y_train)

'''
=================================================
                   GridSearchCV
=================================================
'''
c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000, 10000]
gamma = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# create grid

param_grid = {'C': c, 'gamma': gamma}
grid_search = GridSearchCV(svm.SVC(probability=True), param_grid=param_grid, cv=10, n_jobs=-1)
start = time()
svc_model = grid_search.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(svc_model.cv_results_['params'])))

f = open(filename, "w")
f.write('SVC Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%svc_model.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = svc_model.cv_results_['mean_test_score']
stds = svc_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svc_model.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

best_results_grid = report(svc_model.cv_results_)
print("SVC GridSearchCV Best Parameters: ", svc_model.best_params_)
print("\n GridSearchCV Best Estimator on Training set: ", svc_model.best_estimator_,"\n")

print("=========Evaluation Results on Test Set========\n")
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = svc_model.predict(X_test)
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
plt.title('SVC GridSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/svc/svc_grid.png')
print("Classification report on Test set\n",classification_report(y_test, y_pred))

y_score = svc_model.predict_proba(X_test)[:, 1]

# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('SVC GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/svc/svc_roc.png')
#plt.show()
print("===================================================================================")
'''
==================================================
               RandomizedSearchCV
==================================================
'''
import random

#second split for a random subset for randomized search
X_train_rand, z, y_train_rand, w = train_test_split(X_train, y_train,
                                                    test_size=0.3, random_state=45)
c = []
for i in range(0, 40):
    c.append(random.uniform(0.0001, 1001))
gamma = []
for i in range(0,20):
    gamma.append(random.uniform(0.00001, 0.0001))

print('=======================RANDOM SEARCH================================')
param_dist = {'C': c, 'gamma':gamma}
random_search = RandomizedSearchCV(svm.SVC(probability=True), param_distributions=param_dist, cv=10, n_jobs=-1)
start = time()
svc_rand = random_search.fit(X_train_rand, y_train_rand)
end = time() - start
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(svc_rand.cv_results_['params'])))

f = open(filename, "a")
f.write('SVC Random Search Results\n\n')
f.write("Best parameters set found on development subset:")
f.write('\n\n')
f.write('%s'%svc_rand.best_params_)
f.write('\n\n')
f.write("Grid scores on development subset:")
f.write('\n\n')
means = svc_rand.cv_results_['mean_test_score']
stds = svc_rand.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svc_rand.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

best_results_random = report(svc_rand.cv_results_)
print("SVC RandomSearchCV Best Parameters: ", svc_rand.best_params_)
print("\n Random Search Best Estimator on Random Training Subset: ", svc_rand.best_estimator_,"\n")

print("=========Evaluation Results on Test Set========\n")
y_pred = svc_rand.predict(X_test)
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
sns.heatmap(confusion2, annot=True, annot_kws={"size": 20}, fmt='g')
plt.title('SVC RandomizedSearchCV Confusion Matrix', fontsize = 20)
plt.savefig('results/svc/svc_random.png')
print("Classification Report on Test set\n",classification_report(y_test, y_pred))

# Use the probabilities for your ROC and Precision-recall curves
y_score = svc_rand.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('SVC RandomizedSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/svc/svc_random_roc.png')
print("===================================================================================")
'''
=================================================
              Final GridSearchCV
#=================================================
'''
c = []
gamma = []

''' na vgalw apo sxolio an thelw na dinw kai tis best times apo to prwto grid search
for i in range(0, len(best_results_grid)):
    c.append(best_results_grid[i]['C'])
    gamma.append(best_results_grid[i]['gamma'])
'''
for i in range(0, len(best_results_random)):
    c.append(best_results_random[i]['C'])
    gamma.append(best_results_random[i]['gamma'])
c = list(set(c))
gamma = list(set(gamma))

grid = {'C': c, 'gamma': gamma}

grid_search_final = GridSearchCV(svm.SVC(probability=True), param_grid=grid, cv=10, n_jobs=-1)
start = time()
svc_final = grid_search_final.fit(X_train, y_train)
end = time() - start
print("GridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (end, len(svc_final.cv_results_['params'])))

f = open(filename, "a")
f.write('SVC Final Grid Search Results\n\n')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write('%s'%svc_final.best_params_)
f.write('\n\n')
f.write("Grid scores on development set:")
f.write('\n\n')
means = svc_final.cv_results_['mean_test_score']
stds = svc_final.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svc_final.cv_results_['params']):
   f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
f.close()

report(svc_final.cv_results_)
print("\n Grid Search Best Estimator on Training Subset: ", grid_search_final.best_estimator_,"\n")

best_model = svc_final.best_estimator_
#best_score = best_model.score(X_test,y_test) einai allo score apo to score pou xrhsimopoiw sta upoloipa
#print("Best score=",best_score,"\n")
svc_final.best_params_

print("=========Evaluation Results on Test Set========\n")
y_pred = svc_final.predict(X_test)
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
sns.heatmap(confusion3, annot=True, annot_kws={"size": 20}, fmt='g')
plt.title('SVC GridSearchCV Final Confusion Matrix', fontsize = 20)
plt.savefig('results/svc/svc_final.png')
print("Classification Report on Test Set",classification_report(y_test, y_pred))


y_score = svc_final.predict_proba(X_test)[:, 1]
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)
plt.figure(figsize=(20,10))
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.plot(fpr, tpr, label="auc = "+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('SVC Final GridSearchCV ROC Curve', fontsize = 20)
plt.savefig('results/svc/svc_final_roc.png')
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
gridBest_accuracy = modelEvaluation(filename, "GridSearch Best Model", svc_model, svc_model.best_params_,  X_test, y_test)
randomBest_accuracy = modelEvaluation(filename, "RandomizedSearch Best Model", svc_rand, svc_rand.best_params_,  X_test, y_test)
gridSearch_accuracy = modelEvaluation(filename, "GridSearch Best Model", svc_final, svc_final.best_params_, X_test, y_test)
f = open(filename, "a")
f.write("=======================================================\n")
f.close()