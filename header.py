import numpy as np
import operator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# Utility function to report best scores
def report(results, n_top=5):
    best_results = []
    for i in range(1, n_top + 1):
        print("to i = ",i)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            best_results.append(results['params'][candidate])
    return best_results

'''
def bestResults(cvresults,n_best):
    print("mphka bestResults")
    results = {}
    params = cvresults['params']
    mts = cvresults['mean_test_score']
    j=0
    print("eimai edw")
    for i in params:
        print(i)
        print("to j einai: ",j)
        for k,v in i.items():
            print("to  v einai: ",v)
            results[v] = mts[j]
            j+=1
    print("meta th for")
    best = dict(sorted(results.items(), key=operator.itemgetter(1), reverse=True)[:n_best])
    return best
'''
def bestResults(cvresults):
    print("mphka bestResults")
    results = []
    params = cvresults['params']
    mts = cvresults['mean_test_score']
    j = 0
    print("eimai edw")
    for i in params:
        print("1 for ", i.items())
        for key1, value1 in i.items():
            print("2 for me key %s kai value %s", key1, value1)
            tmp = [key1, value1, mts[j]]
            print(tmp)
            results.append(tmp)
        j += 1

    results.sort(reverse = True)
    return results

def modelEvaluation(filename, model_name, model, bestParam, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f = open(filename, "a")
    f.write('%s'%model_name)
    f.write(' is:\n\t %s\n'%model)
    f.write('Best Parameters %s\n'%bestParam)
    f.write('Accuracy = %s\n\n'%accuracy)
    f.close()
    #plot_confusion_matrix(test_features, predictions)
    return accuracy

def writeResultsToFile(classifier, filename, search, params, results, n_top = 3):
    f = open(filename,"a")
    f.write("=======================%s================================\n"%search)
    f.write("Classifier: %s\n\n" % classifier)
    f.write("Tuning parameters: %s\n"%params)
    f.write("-------------REPORT-------------\n")
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            f.write("Model with rank: {0}\n".format(i))
            f.write("Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            f.write("Parameters: {0}\n\n".format(results['params'][candidate]))

    f.write("=====================================================================\n\n")
    f.close()

def datasetDESCRtoFile(filename, dataset):
    f = open(filename, "w")
    f.write(dataset.DESCR)
    f.write("\nSample counts per class:\n{}\n\n".format(
        {n: v for n, v in zip(dataset.target_names, np.bincount(dataset.target))}))
    f.close()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax