import sklearn.model_selection
import sklearn.datasets as datasets
import sklearn.metrics
import autosklearn.classification
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def example1():
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600,
        per_run_time_limit=300,
        tmp_folder='/tmp/autosklearn_sequential_example_tmp',
        output_folder='/tmp/autosklearn_sequential_example_out',
        # Do not construct ensembles in parallel to avoid using more than one
        # core at a time. The ensemble will be constructed after auto-sklearn
        # finished fitting all machine learning models.
        ensemble_size=0,
        delete_tmp_folder_after_terminate=False,
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')
    # This call to fit_ensemble uses all models trained in the previous call
    # to fit to build an ensemble which can be used with automl.predict()
    automl.fit_ensemble(y_train, ensemble_size=50)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


def example2():
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    print('Data:', X.shape)
    print('Target:', y.shape)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                test_size=0.3, random_state=45)
    import autosklearn.classification
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    automl.cv_results_
    automl.sprint_statistics()
    automl.show_models()


def example3():
    print('Mpika exmample 3')
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    print('Data:', X.shape)
    print('Target:', y.shape)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)

    print('Accuracy score', sklearn.metrics.accuracy_score(y_test, predictions))


example3()
