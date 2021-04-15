# -*- coding: utf-8 -*-
# @Time    : 4/14/21 7:31 PM
# @Author  : Yan
# @Site    : 
# @File    : additional.py
# @Software: PyCharm
import argparse
import numpy as np
import pandas as pd
import time

from numba import njit
from sklearn.linear_model import RidgeClassifierCV

from rocket_function import generate_kernels, apply_kernels, apply_kernel
from utils import create_parser

dataset_names_additional = \
    (
        "ACSF1",
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "BME",
        "Chinatown",
        "Crop",
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "HouseTwenty",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "MelbournePedestrian",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "PLAID",
        "PickupGestureWiimoteZ",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PowerCons",
        "Rock",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShakeGestureWiimoteZ",
        "SmoothSubspace",
        "UMD"
    )


@njit(parallel=True, fastmath=True)
def apply_kernels_jagged(X, kernels, input_lengths):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples = len(X)
    num_kernels = len(weights)

    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2))  # 2 features per kernel

    for i in range(num_examples):

        for j in range(num_kernels):

            # skip incompatible kernels (effective length is "too big" without padding)
            if (input_lengths[i] + (2 * paddings[j])) > ((lengths[j] - 1) * dilations[j]):
                _X[i, (j * 2):((j * 2) + 2)] = \
                    apply_kernel(X[i][:input_lengths[i]], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j],
                                 paddings[j])

    return _X


def run_additional(training_data, test_data, num_runs=10, num_kernels=10_000):
    # assumes variable length time series are padded with nan
    get_input_lengths = lambda X: X.shape[1] - (~np.isnan(np.flip(X, axis=1))).argmax(1)

    def rescale(X, reference_length):
        _X = np.zeros([len(X), reference_length])
        input_lengths = get_input_lengths(X)
        for i in range(len(X)):
            _X[i] = np.interp(np.linspace(0, 1, reference_length), np.linspace(0, 1, input_lengths[i]),
                              X[i][:input_lengths[i]])
        return _X

    def interpolate_nan(X):
        _X = X.copy()
        good = ~np.isnan(X)
        for i in np.where(np.any(~good, 1))[0]:
            _X[i] = np.interp(np.arange(len(X[i])), np.where(good[i])[0], X[i][good[i]])
        return _X

    results = np.zeros(num_runs)
    timings = np.zeros([4, num_runs])  # training transform, test transform, training, test

    Y_training, X_training = training_data[:, 0].astype(np.int), training_data[:, 1:]
    Y_test, X_test = test_data[:, 0].astype(np.int), test_data[:, 1:]

    variable_lengths = False

    # handle three cases: (1) same lengths, no missing values; (2) same lengths,
    # missing values; and (3) variable lengths, no missing values
    # Check whether contains missing values nan
    if np.any(np.isnan(X_training)):

        input_lengths_training = get_input_lengths(X_training)
        input_lengths_training_max = input_lengths_training.max()
        input_lengths_test = get_input_lengths(X_test)

        # missing values (same lengths)
        if np.all(input_lengths_training == input_lengths_training_max):

            X_training = interpolate_nan(X_training)
            X_test = interpolate_nan(X_test)

        # variable lengths (no missing values)
        else:

            variable_lengths = True
            num_folds = 10
            cross_validation_results = np.zeros([2, num_folds])

    # normalise time series
    X_training = (X_training - np.nanmean(X_training, axis=1, keepdims=True)) / (
            np.nanstd(X_training, axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - np.nanmean(X_test, axis=1, keepdims=True)) / (np.nanstd(X_test, axis=1, keepdims=True) + 1e-8)

    for i in range(num_runs):

        # -- variable lengths --------------------------------------------------

        if variable_lengths:

            kernels = generate_kernels(input_lengths_training_max, num_kernels)

            time_a = time.perf_counter()
            X_training_transform_rescale = apply_kernels(rescale(X_training, input_lengths_training_max), kernels)
            X_training_transform_jagged = apply_kernels_jagged(X_training, kernels, input_lengths_training)
            time_b = time.perf_counter()
            timings[0, i] = time_b - time_a

            # indices for cross-validation folds
            I = np.random.permutation(len(X_training))
            I = np.array_split(I, num_folds)

            time_a = time.perf_counter()

            # j = 0 -> rescale
            # j = 1 -> "as is" ("jagged")
            for j in range(2):

                for k in range(num_folds):

                    VA, *TR = np.roll(I, k, axis=0)
                    TR = np.concatenate(TR)

                    classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)

                    if j == 0:  # rescale

                        classifier.fit(X_training_transform_rescale[TR], Y_training[TR])
                        cross_validation_results[j][k] = classifier.score(X_training_transform_rescale[VA],
                                                                          Y_training[VA])

                    elif j == 1:  # jagged

                        classifier.fit(X_training_transform_jagged[TR], Y_training[TR])
                        cross_validation_results[j][k] = classifier.score(X_training_transform_jagged[VA],
                                                                          Y_training[VA])

            best = cross_validation_results.sum(1).argmax()
            time_b = time.perf_counter()
            timings[2, i] = time_b - time_a

            classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)

            if best == 0:  # rescale

                time_a = time.perf_counter()
                X_test_transform_rescale = apply_kernels(rescale(X_test, input_lengths_training_max), kernels)
                time_b = time.perf_counter()
                timings[1, i] = time_b - time_a

                time_a = time.perf_counter()
                classifier.fit(X_training_transform_rescale, Y_training)
                time_b = time.perf_counter()
                timings[2, i] += time_b - time_a

                time_a = time.perf_counter()
                results[i] = classifier.score(X_test_transform_rescale, Y_test)
                time_b = time.perf_counter()
                timings[3, i] = time_b - time_a

            elif best == 1:  # jagged

                time_a = time.perf_counter()
                X_test_transform_jagged = apply_kernels_jagged(X_test, kernels, input_lengths_test)
                time_b = time.perf_counter()
                timings[1, i] = time_b - time_a

                time_a = time.perf_counter()
                classifier.fit(X_training_transform_jagged, Y_training)
                time_b = time.perf_counter()
                timings[2, i] += time_b - time_a

                time_a = time.perf_counter()
                results[i] = classifier.score(X_test_transform_jagged, Y_test)
                time_b = time.perf_counter()
                timings[3, i] = time_b - time_a

        # -- same lengths ------------------------------------------------------

        else:

            kernels = generate_kernels(X_training.shape[1], num_kernels)

            # -- transform training --------------------------------------------

            time_a = time.perf_counter()
            X_training_transform = apply_kernels(X_training, kernels)
            time_b = time.perf_counter()
            timings[0, i] = time_b - time_a

            # -- transform test ------------------------------------------------

            time_a = time.perf_counter()
            X_test_transform = apply_kernels(X_test, kernels)
            time_b = time.perf_counter()
            timings[1, i] = time_b - time_a

            # -- training ------------------------------------------------------

            time_a = time.perf_counter()
            classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)
            classifier.fit(X_training_transform, Y_training)
            time_b = time.perf_counter()
            timings[2, i] = time_b - time_a

            # -- test ----------------------------------------------------------

            time_a = time.perf_counter()
            results[i] = classifier.score(X_test_transform, Y_test)
            time_b = time.perf_counter()
            timings[3, i] = time_b - time_a

    return results, timings


if __name__ == '__main__':
    arguments = create_parser()
    # == run through the additional datasets =======================================
    results_additional = pd.DataFrame(index=dataset_names_additional,
                                      columns=["accuracy_mean",
                                               "accuracy_standard_deviation",
                                               "time_training_seconds",
                                               "time_test_seconds"],
                                      data=0)
    results_additional.index.name = "dataset"

    compiled = False

    print(f"RUNNING".center(80, "="))

    for dataset_name in dataset_names_additional:

        print(f"{dataset_name}".center(80, "-"))

        # -- read data -------------------------------------------------------------

        print(f"Loading data".ljust(80 - 5, "."), end="", flush=True)

        if dataset_name != "PLAID":

            training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv")
            test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.tsv")

        else:

            training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv", delimiter=",")
            test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.tsv", delimiter=",")

        print("Done.")

        # -- precompile ------------------------------------------------------------

        if not compiled:
            print(f"Compiling ROCKET functions (once only)".ljust(80 - 5, "."), end="", flush=True)

            _ = generate_kernels(100, 10)
            apply_kernels(np.zeros_like(training_data)[:, 1:], _)
            apply_kernels_jagged(np.zeros_like(training_data)[:, 1:], _,
                                 np.array([training_data.shape[1]] * len(training_data)))
            compiled = True

            print("Done.")

        # -- run -------------------------------------------------------------------

        print(f"Performing runs".ljust(80 - 5, "."), end="", flush=True)

        results, timings = run_additional(training_data, test_data,
                                          num_runs=arguments.num_runs,
                                          num_kernels=arguments.num_kernels)
        timings_mean = timings.mean(1)

        print("Done.")

        # -- store results ---------------------------------------------------------

        results_additional.loc[dataset_name, "accuracy_mean"] = results.mean()
        results_additional.loc[dataset_name, "accuracy_standard_deviation"] = results.std()
        results_additional.loc[dataset_name, "time_training_seconds"] = timings_mean[[0, 2]].sum()
        results_additional.loc[dataset_name, "time_test_seconds"] = timings_mean[[1, 3]].sum()

    print(f"FINISHED".center(80, "="))

    results_additional.to_csv(f"{arguments.output_path}/results_additional.csv")
