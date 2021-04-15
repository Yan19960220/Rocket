# -*- coding: utf-8 -*-
# @Time    : 3/21/21 9:16 PM
# @Author  : Yan
# @Site    : 
# @File    : data.py
# @Software: PyCharm

import csv
import random
import numpy as np
from pandas.core.frame import DataFrame
from utils import file2list

VALUES_FILE = './data/glitch_values.bin'
TIMES_FILE = './data/glitch_times.bin'
LENGTHS_FILE = './data/glitch_lengths.bin'
METADATA_FILE = './data/trainingset_v1d1_metadata.csv'
POS_FILE = './pos.txt'

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


def calculate_time(time, time_ns):
    return np.float64(time + '.' + '0' * (9 - len(time_ns)) + time_ns)


def not_contain_nan(np_array):
    if type(np_array) is not np.ndarray:
        return True not in np.isnan(list2array(np_array))
    else:
        return True not in np.isnan(np_array)


def getData(random_range=[50], DURATION_TO_EXAMINE=0.5):
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
        result[item] = []

    for item_range in random_range:
        for duration in HALF_VALUES_PER_DURATION:
            segment_per_class = get_segment_per_class(duration, glitch_classes_inverted, glitch_peak,
                                                      glitch_times, glitch_values)
            result[item_range].append(random_sample(segment_per_class, item_range))
        save2txt(item_range, result)
        print(">> time_series range - " + str(item_range))
    return result


def save2txt(item_range, result):
    source = result[item_range][0]
    data = DataFrame(source, index=None).set_index(0)
    data.to_csv("./data/" + str(item_range) + ".csv")


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
            if len(segment_per_class[k]) > sample_range:
                list_a = random.sample(segment_per_class[k], sample_range)
                array = list2array(list_a)
            dataset[k] = (array[:, 0].tolist(), array[:, 1].tolist())
    dataset = dataSet(dataset)
    ts = []
    for label in dataset:
        (list_time_series, list_indexes) = dataset[label]
        for time_series, indexes in zip(list_time_series, list_indexes):
            time_series.insert(0, data_index[label])
            ts.append(time_series)
    return ts


def load_matrix_from_file(matrix_filepath, length):
    return np.fromfile(matrix_filepath).reshape([length, length])


def list2array(listA):
    return np.array(listA)


def dataSet(random_segments):
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
