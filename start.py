#!/usr/bin/python3

import pickle
import time
import numpy as np
import scipy.special
from tqdm import tqdm

import svm
from levenshtein import edit_kernel
from spectrum import k_spectrum_mismatch, k_spectrum
from utils import precomputed_kernels, svm_kernels, transform_kernels

# TODO: CYTHON setup.py?
# TODO: README?


def get_kernels(spectrum_k=39, mismatch_k=12, use_edit_kernel=True):
    kernels = dict()

    # Spectrum kernels
    for k in tqdm(range(1, spectrum_k+1), 'Computing Spectrum kernels'):
        name = 'spectrum_{}'.format(k)
        kernels[name] = (precomputed_kernels, dict(kernel=k_spectrum, name=name, k=k))

    # Mismatch kernels
    for k in tqdm(range(1, mismatch_k+1), 'Computing Mismatch kernels'):
        name = 'mismatch_1_{}'.format(k)
        kernels[name] = (precomputed_kernels, dict(kernel=k_spectrum_mismatch, name=name, k=k, decay=1))

    # Edit kernel
    if use_edit_kernel:
        print('Computing Edit kernel')
        kernels['edit_gaussian'] = (edit_kernel, dict(kernel='gaussian', scale=3.25))

    return kernels


def separate_evaluation(kernels, file='separate_kernels', repeats=3, model=svm.SVCCoordinate, **params):
    try:
        with open(file, 'rb') as src:
            results = pickle.load(src)
    except FileNotFoundError:
        results = dict()

    print('Evaluating single kernels...')
    t0 = time.time()
    for kernel in kernels:
        if kernel in results:
            print("Skipping kernel {}, results already computed.".format(kernel))
            continue

        print("Evaluating kernel {}:".format(kernel))
        t = time.time()

        # Kernel function and parameters.
        kernel_f, kernel_params = kernels[kernel]

        # Evaluates the kernel.
        results[kernel] = svm_kernels(kernel_f(**kernel_params), model=model, repeats=repeats, **params)

        # Saves the results.
        with open(file, 'wb') as dst:
            pickle.dump(results, dst)
        print("Duration {:.1f}s\n".format(time.time() - t))

    print("Total time: {:.1f}s".format(time.time() - t0))


def weighted_kernel(kernels, file='separate_kernels', Ts=(1, 1, 1), out_weights_file=None):
    Ts = np.array(Ts)

    with open(file, 'rb') as src:
        results = pickle.load(src)

    weights = []
    kernel_names = []
    for kernel in kernels:
        if kernel not in results:
            raise ValueError("Kernel {} not found in results.".format(kernel))
        weights.append(np.array(results[kernel])[:, 0])
        kernel_names.append(kernel)

    weights = np.array(weights) * Ts[None, :]
    weights -= scipy.special.logsumexp(weights, axis=0)
    weights = np.exp(weights)

    Ks = None
    for w, kernel in zip(weights, kernel_names):
        kernel_f, kernel_params = kernels[kernel]
        new_Ks = kernel_f(**kernel_params)

        if Ks is None:
            Ks = transform_kernels([new_Ks], lambda j, K: w[j] * K)
        else:
            Ks = transform_kernels([Ks, new_Ks], lambda j, K1, K2: K1 + w[j] * K2)

    if out_weights_file is not None:
        with open(out_weights_file, 'w') as dst:
            dst.write("kernel_name: weight_0 weight_1 weight_2\n")

            for w, kernel in zip(weights, kernel_names):
                dst.write("{} {:e} {:e} {:e}\n".format(kernel, *tuple(w)))

    return Ks


def optimize_T(kernels, file='separate_kernels', repeats=4, model=svm.SVCCoordinate, Ts=np.linspace(0, 200, 6),
               **params):
    results = []
    for T in Ts:
        print("T = {}".format(T))
        t = time.time()
        results.append(
            svm_kernels(kernels=weighted_kernel(kernels, file=file, Ts=(T, T, T)),
                        model=model, repeats=repeats,
                        **params))
        print("Duration {:.2f}s".format(time.time() - t))

    results = np.array(results)
    return Ts[np.argmax(results[:, :, 0], axis=0)]


def final_submission(spectrum_k=39, mismatch_k=12, use_edit_kernel=True, compute_T=False,
                     file='separate_kernels', out_weights_file=None,
                     **params):

    # Gathers all possible kernels.
    kernels = get_kernels(spectrum_k, mismatch_k, use_edit_kernel)

    # Evaluates single kernels.
    separate_evaluation(kernels, file=file, **params)

    # Optimizes the T parameter.
    if compute_T:
        print('Optimizing T values...')
        Ts = optimize_T(kernels, file=file, **params)
        print("Best values for T: {}, {}, {}.".format(*tuple(Ts)))
    else:
        # Uses already computed values.
        Ts = (40., 120., 120.)
        print('Using pre-defined values for T: {}, {}, {}'.format(*Ts))

    # Computes final kernel.
    final_kernel = weighted_kernel(kernels, file, Ts, out_weights_file)

    # Evaluates and generates final submission file.
    print('\nEvaluating final submission...')
    if 'repeats' not in params:
        params['repeats'] = 4
    res = svm_kernels(final_kernel, prediction_file='Yte.csv', **params)
    print('Submission file created.')
    return res


if __name__ == '__main__':
    final_submission(spectrum_k=39,
                     mismatch_k=12,
                     use_edit_kernel=True,

                     compute_T=False,

                     file='separate_kernels',
                     out_weights_file='out_weights',

                     model=svm.SVCCoordinate,
                     intercept=1.,
                     loss='hinge')
