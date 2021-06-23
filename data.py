# -*- coding: utf-8 -*-
# @Time    : 3/21/21 9:16 PM
# @Author  : Yan
# @Site    : 
# @File    : data.py
# @Software: PyCharm
import os
import csv
import random
import numpy as np
import scipy
from matplotlib import mlab
from pandas.core.frame import DataFrame
from scipy.interpolate import interp1d
from scipy.stats import stats

from utils import file2list, check_if_file_exits, whiten, bandpass, zca_whitening, delete_history

BASE_DIR = '../virgo_data/'
VALUES_FILE = BASE_DIR + './data/glitch_values.bin'
TIMES_FILE = BASE_DIR + './data/glitch_times.bin'
LENGTHS_FILE = BASE_DIR + './data/glitch_lengths.bin'
METADATA_FILE = BASE_DIR + './data/trainingset_v1d1_metadata.csv'
POS_FILE = './pos.txt'
OUTPUT_DATA = './virgo/'

OUTPUT_POS = True
load_history_matrix = False

data_index = {
    'Air_Compressor': 0,
    '1400Ripples': 1,
    '1080Lines': 2,
    'Blip': 3,
    'Extremely_Loud': 4,
    'Koi_Fish': 5,
    'Chirp': 6,
    'Light_Modulation': 7,
    'Low_Frequency_Burst': 8,
    'Low_Frequency_Lines': 9
}


def intercept_part_dataSet(random_segments):
    return {'Air_Compressor': random_segments['Air_Compressor'],  # (50, 10)
            '1400Ripples': random_segments['1400Ripples'],
            '1080Lines': random_segments['1080Lines'],
            'Blip': random_segments['Blip'],
            'Extremely_Loud': random_segments['Extremely_Loud'],
            'Koi_Fish': random_segments['Koi_Fish'],
            'Chirp': random_segments['Chirp'],
            'Light_Modulation': random_segments['Light_Modulation'],
            'Low_Frequency_Burst': random_segments['Low_Frequency_Burst'],
            'Low_Frequency_Lines': random_segments['Low_Frequency_Lines']}


def calculate_time(time, time_ns):
    return np.float64(time + '.' + '0' * (9 - len(time_ns)) + time_ns)


def not_contain_nan(np_array):
    if type(np_array) is not np.ndarray:
        return True not in np.isnan(list2array(np_array))
    else:
        return True not in np.isnan(np_array)


def preprocessing(X, phase_shift=0, time_shift=0):
    fband = [35.0, 350.0]
    T = 800.0
    XWhiten = whiten(X, dt=T)
    return bandpass(XWhiten, fband, T)


def getData(random_range=[50], DURATION_TO_EXAMINE=0.5, is_split=True):
    # Type Check
    if not isinstance(random_range, list):
        raise TypeError('random_range is not type List: {}'.format(type(random_range)))

    glitch_classes_inverted, glitch_peak_times, glitch_times, glitch_values = read_data()

    glitch_values = list2array(glitch_values)
    glitch_times = list2array(glitch_times)
    glitch_peak = list2array(glitch_peak_times)

    FREQUENCY = 4096
    HALF_VALUES_PER_DURATION = []
    if isinstance(DURATION_TO_EXAMINE, float):
        HALF_VALUES_PER_DURATION = [int(DURATION_TO_EXAMINE * FREQUENCY / 2)]
    elif isinstance(DURATION_TO_EXAMINE, list):
        HALF_VALUES_PER_DURATION = [int(duration * FREQUENCY / 2) for duration in DURATION_TO_EXAMINE]

    result = {}

    for item in random_range:
        result[item] = {}

    split_rate = [0.6, 0.4]
    for item_range in random_range:
        for duration in HALF_VALUES_PER_DURATION:
            segment_per_class = get_segment_per_class(duration, glitch_classes_inverted, glitch_peak,
                                                      glitch_times, glitch_values)
            established_data = random_sample(segment_per_class, item_range)
            result[item_range][duration] = established_data
            # split_num = [int(num*result[item_range][0].__len__()) for num in split_rate]

            suffix = '_' + str(duration)

            if is_split:
                print(f"Begin split data".center(80, "-"))
                delete_history(["./data/" + str(item_range) + suffix + "_TRAIN" + ".csv",
                                "./data/" + str(item_range) + suffix + '_TEST' + ".csv"])
                refactor_data = {}
                train_data = []
                test_data = []
                for label in range(data_index.__len__()):
                    refactor_data[label] = []
                    for item in result[item_range][duration]:
                        if item[0] == label:
                            refactor_data[label].append(item)
                    length_for_label = refactor_data[label].__len__()
                    train_data_num = int(length_for_label*split_rate[0])

                    train_data += refactor_data[label][:train_data_num]
                    test_data += refactor_data[label][train_data_num:]

                    # train_data = result[item_range][0][:split_num[0]]
                    # test_data = result[item_range][0][split_num[0]:]

                save2txt(train_data, item_range, suffix + '_TRAIN')
                save2txt(test_data, item_range, suffix + '_TEST')
            else:
                delete_history("./data/" + str(item_range) + suffix + ".csv")
                save2txt(result, item_range, None, duration)
            print(">> time_series range - " + str(item_range))
    print(f"finished all".center(90, '*'))
    return result


def save2csv_dict(dict_data, item_range, data_type):
    my_list = [dict_data]
    data = DataFrame(my_list, index=None).set_index(0)
    data.to_csv(OUTPUT_DATA + str(item_range) + data_type + ".csv")


def save2txt(data, item_range, data_type, duration=None):
    if data_type is None:
        source = data[item_range][duration]
        data = DataFrame(source, index=None).set_index(0)
        data.to_csv(OUTPUT_DATA + str(item_range) + '_' + str(duration) + ".csv")
    else:
        source = data
        data = DataFrame(source, index=None).set_index(0)
        data.to_csv(OUTPUT_DATA + str(item_range) + data_type + ".csv", mode='a', index=True)
    print(f"save data finished".center(80, "-"))


def read_data():
    values = np.fromfile(VALUES_FILE, dtype=np.float64)
    times = np.fromfile(TIMES_FILE, dtype=np.float64)
    lengths = np.fromfile(LENGTHS_FILE, dtype=np.int32)

    durations, metadata = extra_data_from_meta(METADATA_FILE)

    glitch_classes_inverted = {}
    glitch_classes = []
    glitch_values = []
    glitch_times = []
    glitch_peak_times = []
    current_offset = 0
    for i, meta_entry in enumerate(metadata):

        if i >= len(lengths):
            break

        current_length = lengths[i]

        if current_length == 0:
            continue

        if meta_entry[4] not in glitch_classes_inverted:
            glitch_classes_inverted[meta_entry[4]] = []

        glitch_classes.append(meta_entry[4])  # label
        glitch_peak_times.append(meta_entry[1])  # peak time
        glitch_values.append(values[current_offset: current_offset + current_length])
        glitch_times.append(times[current_offset: current_offset + current_length])
        glitch_classes_inverted[meta_entry[4]].append(len(glitch_peak_times))
        current_offset += current_length
    for glitch_class in glitch_classes_inverted:
        glitch_classes_inverted[glitch_class] = list2array(glitch_classes_inverted[glitch_class])
    return glitch_classes_inverted, glitch_peak_times, glitch_times, glitch_values


def get_segment_per_class(duration, glitch_classes_inverted, glitch_peak, glitch_times, glitch_values):
    glitch_segments = []
    for current_values, current_times, current_peak in zip(glitch_values, glitch_times, glitch_peak):
        peak_indices = np.searchsorted(current_times, current_peak)
        glitch_segments.append(
            current_values[peak_indices - duration: peak_indices + duration]
        )
    glitch_segments = list2array(glitch_segments)
    segment_per_class = {}
    for glitch_class, glitch_ids in glitch_classes_inverted.items():
        if len(glitch_ids) > 1:
            segment_per_class[glitch_class] = []

            for i in range(len(glitch_ids) - 1):
                if not_contain_nan(glitch_segments[glitch_ids[i]]):
                    segment_per_class[glitch_class].append([glitch_segments[glitch_ids[i]].tolist(), glitch_ids[i]])
    return segment_per_class


def random_sample(segment_per_class, sample_range):
    """ Returns the list of the random sample
               (new_indexes, time_series, label)

        Argument
        ---------
        segment_per_class
               The list of time_series
               The list of the corresponding indexes
    """
    dataset = {}
    poses = []
    if load_history_matrix:
        poses = file2list(POS_FILE)
    if poses:
        for k in segment_per_class.keys():
            dataset[k] = ([], [])
            for item in segment_per_class[k]:
                if item[1] in poses:
                    dataset[k][0].append(item[0])
                    dataset[k][1].append(item[1])
    else:
        for k in segment_per_class.keys():
            array = list2array(segment_per_class[k])
            if sample_range > 0:
                if len(segment_per_class[k]) > sample_range:
                    list_a = random.sample(segment_per_class[k], sample_range)
                    array = list2array(list_a)
            dataset[k] = (array[:, 0].tolist(), array[:, 1].tolist())
    dataset = intercept_part_dataSet(dataset)
    ts = []
    for label in dataset:
        (list_time_series, list_indexes) = dataset[label]
        for time_series, indexes in zip(list_time_series, list_indexes):
            # time_series = preprocessing(time_series)
            time_series.insert(0, data_index[label])
            ts.append(time_series)
    return ts


def load_matrix_from_file(matrix_filepath, length):
    return np.fromfile(matrix_filepath).reshape([length, length])


def list2array(listA):
    return np.array(listA)


def extra_data_from_meta(file):
    metadata = []
    durations = []
    with open(file, 'r') as metadata_file:
        metadata_reader = csv.reader(metadata_file, delimiter=',')
        metadata_header = next(metadata_reader)

        metadata = [[entry[1],  # ifo
                     calculate_time(entry[2], entry[3]),  # peak time
                     calculate_time(entry[4], entry[5]),  # start time
                     np.float64(entry[6]),  # duration
                     entry[22]]  # label
                    for entry in metadata_reader]
        durations = [item[-1] for item in metadata]
    return durations, metadata
