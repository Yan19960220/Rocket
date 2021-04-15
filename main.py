from sklearn.linear_model import RidgeClassifierCV

from data import getData
from rocket_function import generate_kernels, apply_kernels
from utils import create_parser, normalization, standardization
import pandas as pd
import numpy as np
import time


def random_sel_exp(max_num: int, min_num: int = 1):
    while min_num <= max_num and np.random.randint(2) == 1:
        min_num += 1
    return min_num


def update_data():
    range_list = [50, 100, 200, 300]
    _ = getData(random_range=range_list)


def run(training_data, test_data, num_runs=10, num_kernels=10_000):
    results = np.zeros(num_runs)
    timings = np.zeros([4, num_runs])  # training transform, test transform, training, test

    Y_training, X_training = training_data[:, 0].astype(int), standardization(normalization(training_data[:, 1:]))
    Y_test, X_test = test_data[:, 0].astype(int), standardization(normalization(test_data[:, 1:]))

    for i in range(num_runs):
        input_length = X_training.shape[1]
        kernels = generate_kernels(input_length, num_kernels)

        # -- transform training ------------------------------------------------

        time_a = time.perf_counter()
        X_training_transform = apply_kernels(X_training, kernels)
        time_b = time.perf_counter()
        timings[0, i] = time_b - time_a

        # -- transform test ----------------------------------------------------

        time_a = time.perf_counter()
        X_test_transform = apply_kernels(X_test, kernels)
        time_b = time.perf_counter()
        timings[1, i] = time_b - time_a

        # -- training ----------------------------------------------------------

        time_a = time.perf_counter()
        classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)
        classifier.fit(X_training_transform, Y_training)
        time_b = time.perf_counter()
        timings[2, i] = time_b - time_a

        # -- test --------------------------------------------------------------

        time_a = time.perf_counter()
        results[i] = classifier.score(X_test_transform, Y_test)
        time_b = time.perf_counter()
        timings[3, i] = time_b - time_a

    return results, timings


if __name__ == '__main__':
    arguments = create_parser()

    if arguments.update_data:
        update_data()

    dataset_range = \
        (
            50,
            100
            # 200,
            # 300
        )
    results_dataset = pd.DataFrame(index=dataset_range,
                                    columns=["accuracy_mean",
                                             "accuracy_standard_deviation",
                                             "time_training_seconds",
                                             "time_test_seconds"],
                                    data=0)
    results_dataset.index.name = "dataset"

    compiled = False

    print(f"RUNNING".center(80, "="))

    for dataset_name in dataset_range:

        print(f"{dataset_name}".center(80, "-"))

        print(f"Loading data".ljust(80 - 5, "."), end="", flush=True)
        training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}.csv", delimiter=',')
        test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}.csv", delimiter=',')
        training_data = np.delete(training_data, 0, axis=0)
        test_data = np.delete(test_data, 0, axis=0)
        print("Done.")

        if not compiled:
            print(f"Compiling ROCKET functions (once only)".ljust(80 - 5, "."), end="", flush=True)

            _ = generate_kernels(100, 10)
            apply_kernels(np.zeros_like(training_data)[:, 1:], _)
            compiled = True

            print("Done.")

        # -- run -------------------------------------------------------------------

        print(f"Performing runs".ljust(80 - 5, "."), end="", flush=True)
        results, timings = run(training_data, test_data,
                               num_runs=arguments.num_runs,
                               num_kernels=arguments.num_kernels)
        timings_mean = timings.mean(1)

        print("Done.")

        # -- store results ---------------------------------------------------------

        results_dataset.loc[dataset_name, "accuracy_mean"] = results.mean()
        results_dataset.loc[dataset_name, "accuracy_standard_deviation"] = results.std()
        results_dataset.loc[dataset_name, "time_training_seconds"] = timings_mean[[0, 2]].sum()
        results_dataset.loc[dataset_name, "time_test_seconds"] = timings_mean[[1, 3]].sum()
    print(f"FINISHED".center(80, "="))
    results_dataset.to_csv(f"{arguments.output_path}/results_dataset.csv")
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    #     device = 'cpu'
    #
    # arguments = create_parser()
    #
    # dataset = {}
    #
    # num_kernels = 1000*10
    # max_len_local_field = 200
    # num_window_sizes = 5
    #
    # candidates = {
    #     'kernel_size': [3, 5, 9, 17, 33, 65],
    #     'stride': 1,
    #
    #     'dilation': None,  # limited to a local neighbourhood
    #     'padding': None,  # always equal-padding
    #
    #     'window_size': None,
    # }
    #
    # no2plot = None
    # # no2plot = set([5])
    #
    # if 'positions' not in globals():
    #     positions = {}
    #
    # with torch.no_grad():
    #     for no, (split, values) in tqdm(dataset.items()):
    #         if no2plot is None or no in no2plot:
    #             if no in positions and len(positions[no]) == num_kernels:
    #                 continue
    #             elif no not in positions:
    #                 positions[no] = []
    #
    #             series_length = len(values)
    #             series_tensor = torch.from_numpy(values.reshape([1, 1, -1])).to(device)
    #
    #             stride = candidates['stride']
    #             window_size_range = (50, 300)
    #             #             onetenth_testrange = int((series_length - split) / 10)
    #             #             window_sizerange = (100 if 100 < onetenth_testrange else onetenth_testrange,
    #             #                                 200 if onetenth_testrange < 200 else onetenth_testrange)
    #
    #             for i in tqdm(range(len(positions[no]), num_kernels)):
    #                 kernel_size = candidates['kernel_size'][np.random.randint(len(candidates['kernel_size']))]
    #                 dilation = random_sel_exp(int(np.floor(np.log2(max_len_local_field / kernel_size))))
    #                 padding = int(((kernel_size - 1) / 2) * dilation)
    #
    #                 local_kernel = torch.nn.Conv1d(1, 1, kernel_size, stride=stride, padding=padding,
    #                                                dilation=dilation, bias=False).float().to(device)
    #                 torch.nn.init.normal_(local_kernel.weight, mean=0.0, std=1.0)
    #                 local_conved_series = local_kernel(series_tensor)
    #
    #                 if device == 'cuda':
    #                     local_conved_series = local_conved_series.detach().cpu().numpy().reshape([-1])
    #                 else:
    #                     local_conved_series = local_conved_series.detach().numpy().reshape([-1])
    #
    #                 window_sizes = np.random.randint(window_size_range[0], window_size_range[1], size=num_window_sizes)
    #
    #                 local_pmp = mp.compute(local_conved_series, windows=window_sizes, n_jobs=32)
    #                 local_discord = mp.discover.discords(local_pmp, k=1)['discords'][0][1]
    #
    #                 centered_discord = int(local_discord + np.mean(window_sizes))
    #                 positions[no].append(centered_discord)

