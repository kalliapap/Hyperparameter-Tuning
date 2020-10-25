from sklearn import model_selection, datasets
from tpot import TPOTClassifier
from time import time

dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target
print('Data:', X.shape)
print('Target:', y.shape)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=45)

'''
=============SVC with TPOT=============
'''
svc_params = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000, 10000],
        'kernel': ['rbf'],
        'gamma': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    }
my_tpot = TPOTClassifier(generations=4, population_size=24, offspring_size=12,
                         verbosity=2, config_dict={'sklearn.svm.SVC': svc_params}, cv=10)

start = time()
my_tpot.fit(X_train, y_train)
end = time() - start
print("SVC with TPOT took %.2f seconds.\n"
      % end)

print(my_tpot.score(X_test, y_test))
print()
my_tpot.export('tpot_exported_pipeline_svc.py')
'''
=============RF with TPOT=============
'''

rf_parameters = {'max_depth': [20, 50, 80, 100, None],'max_features': ['auto'],
                 'min_samples_leaf': [1, 2],'min_samples_split': [2, 5],
                 'n_estimators': [200, 600, 1200, 1800, 2000]}

my_tpot = TPOTClassifier(generations=4, population_size=24, offspring_size=12, verbosity=2,
                         config_dict={'sklearn.ensemble.RandomForestClassifier': rf_parameters}, cv=10)
start = time()
my_tpot.fit(X_train, y_train)
end = time() - start
print("RF with TPOT took %.2f seconds.\n"
      % end)

print(my_tpot.score(X_test, y_test))
print()
my_tpot.export('tpot_exported_pipeline_rf.py')
'''
=============KNN with TPOT=============
'''
knn_params = {'n_neighbors': [1, 5, 10, 15], 'leaf_size': [1, 2, 4, 8], 'p': [1, 2]}

my_tpot = TPOTClassifier(generations=4, population_size=24, offspring_size=12,
                         verbosity=2, config_dict={'sklearn.neighbors.KNeighborsClassifier': knn_params}, cv=10)

start = time()
my_tpot.fit(X_train, y_train)
end = time() - start
print("KNN with TPOT took %.2f seconds.\n"
      % end)

print(my_tpot.score(X_test, y_test))
my_tpot.export('tpot_exported_pipeline_knn.py')
'''
=============Simple TPOT=============
'''
my_tpot = TPOTClassifier(generations=4, population_size=24, verbosity=2, early_stop=12, cv=10)

start = time()
my_tpot.fit(X_train, y_train)
end = time() - start
print("TPOT took %.2f seconds.\n"% end)

print(my_tpot.score(X_test, y_test))
my_tpot.export('tpot_exported_pipeline.py')