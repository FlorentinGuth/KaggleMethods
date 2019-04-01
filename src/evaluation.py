import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor
from data import train_Ys, k_folds_indices, save_predictions


def evaluate(classifier, K, Y, folds=5, repeats=1):
    """
    :param classifier: classifier to evaluate
    :param K: precomputed kernel matrix of shape (n_samples, n_samples)
    :param Y: training labels of shape (n_samples, )
    :param folds: number of folds to use
    :param repeats: number of repetitions of the k-fold evaluation

    Evaluates the classifier using cross-validation.
    Returns the mean and std of the validation and train scores over the repetitions:
        (valid_scores_mean, valid_scores_std, train_scores_mean, train_scores_std)
    """

    N = len(K)
    valid_scores = []
    train_scores = []
    for i in range(repeats):
        valid_score = 0
        train_score = 0
        for train_inds, valid_inds in k_folds_indices(N, folds):
            classifier.fit(K[np.ix_(train_inds, train_inds)], Y[train_inds])

            valid_score += (classifier.predict(K[np.ix_(
                valid_inds, train_inds)]) == Y[valid_inds]).sum()
            train_score += (classifier.predict(K[np.ix_(
                train_inds, train_inds)]) == Y[train_inds]).mean()

        valid_scores.append(valid_score / N)
        train_scores.append(train_score / folds)

    valid_scores = np.array(valid_scores)
    train_scores = np.array(train_scores)

    return valid_scores.mean(), valid_scores.std(), train_scores.mean(), train_scores.std()


def grid_search(model, params, K, Y, folds=5, repeats=1):
    """
     Grid search on params using k-fold cross validation.
    :param model: a function that instantiates a model given parameters from params
    :param params: list of parameters to try
    :param K the kernel
    :param Y the labels
    :param folds number of folds for validation
    :return selected parameters and associated performance
    """
    results = []
    for p in params:
        results.append(np.array(evaluate(model(**p), K, Y, folds=folds, repeats=repeats)))

    results = np.array(results)
    p = params[np.argmax(results[:, 0] - results[:, 1])]

    return p, np.array(evaluate(model(**p), K, Y, folds=20, repeats=repeats))


def final_train(model, p, K_train, Y_train, K_test):
    """
    :param model: a model constructor
    :param p: model parameters
    :param K_train: training kernel
    :param Y_train: training labels
    :param K_test: test kernel
    :return: predictions of the trained model on the test data.
    """
    m = model(**p)
    m.fit(K_train, Y_train)
    return m.predict(K_test)


def svm_kernels(kernels, model, Cs=10. ** np.arange(-3, 4), prediction_file=None, repeats=1, **params):
    """
    Evaluates a SVM model with the specified kernels.
    - First, optimizes C value using a grid search for each of the data sets.
    - Evaluates the performance of the best C .
    - Trains on the full data set and generates test predictions (if prediction_file is not None).

    :param kernels: (train_Ks, test_Ks) containing train and test kernels for all data sets
    :param model: model constructor
    :param Cs: values of C to use in the grid search
    :param prediction_file: file to save the predictions to
    :param repeats: number of repetitions of the k-fold cross validations.
    :param params: parameters of the model (excluding C)
    :return: detailed validation score over each of the 3 data sets.
    """
    train_Ks, test_Ks = kernels

    params = [dict(C=C, **params) for C in Cs]
    total_perf = np.zeros(4)
    full_results = []

    with Executor(max_workers=3) as executor:
        futures = [executor.submit(grid_search, model, params, K, Y, repeats=repeats) for K, Y in
                   zip(train_Ks, train_Ys)]
        res = [future.result() for future in futures]

    for p, performance in res:
        total_perf += performance
        percentages = tuple(100 * performance)
        print('dataset: Validation {:.2f} ± {:.2f}\t Train {:.2f} ± {:.2f}\t C={:.0e}'.format(*percentages, p['C']))
        full_results.append(performance)

    if prediction_file is not None:
        with Executor(max_workers=3) as executor:
            futures = [executor.submit(final_train, model, p, K, Y, K_test) for K, Y, K_test, (p, _) in
                       zip(train_Ks, train_Ys, test_Ks, res)]
            predictions = [future.result() for future in futures]
        save_predictions(predictions, prediction_file)

    total_percentages = 100 * total_perf / 3
    print('total:   Validation {:.2f} ± {:.2f}\t Train {:.2f} ± {:.2f}\t'.format(*total_percentages))
    return full_results
