from collections import defaultdict, deque
import json
import os
import random
import itertools

OBJECT_NAMES = ['key', 'ball', 'box']
COLOR_NAMES = ['blue', 'green', 'yellow', 'grey', 'purple', 'red']


def train_test_splits(train_test_ratio=5, seed=1):
    assert (len(OBJECT_NAMES) >= 2 and len(COLOR_NAMES) >= 2)

    # create all combinations of objects and colours
    train_pairs = list(itertools.product(OBJECT_NAMES, COLOR_NAMES))

    test_pairs = []
    test_size = max(1, len(train_pairs) // (1+train_test_ratio))
    # keep track of colors and objects in the test set to ensure all unique colors and objects are seen
    # before adding a second instance of a color or object in the test pairs
    test_element_counts = {'colors': defaultdict(int), 'objs': defaultdict(int)}

    # shuffle
    random.seed(seed)
    random.shuffle(train_pairs)

    # seed for subsequent sampled sequence of objects and colours
    random.seed(seed)
    while len(test_pairs) < test_size:

        obj, color = random.choice(train_pairs)

        if color not in test_element_counts['colors'] or len(test_element_counts['colors']) >= len(COLOR_NAMES):
            if obj not in test_element_counts['objs'] or len(test_element_counts['objs']) >= len(OBJECT_NAMES):

                test_pairs.append((obj, color))
                test_element_counts['objs'][obj] += 1
                test_element_counts['colors'][color] += 1

                train_pairs.remove((obj, color))

    # augment dataset by duplicating (obj, color) pairs pairs in the dataset of color or objects that were moved to the
    # test set and therefore have lower representation in the train set
    test_objs = deque(test_element_counts['objs'].keys())
    test_colors = deque(test_element_counts['colors'].keys())
    test_objs.rotate()
    for i in range(test_size):
        pair = (test_objs[i], test_colors[i])
        if pair not in test_pairs:
            train_pairs.append(pair)

    return {'seed': seed, 'train_pairs': train_pairs, 'test_pairs': test_pairs}


def save_json(file=None, filename='dataset_pairs', path=None):
    if not os.path.exists(path):
       os.makedirs(path)

    file_path = os.path.join(path, filename +'.json')
    with open(file_path, 'w') as f:
        json.dump(file, f)


def save_txt(file=None, filename='mission_strings', path=None):
    if not os.path.exists(path):
       os.makedirs(path)
    file_path = os.path.join(path, filename + '.txt')
    with open(file_path, 'a') as f:
        for w in file:
            f.write(w)
            f.write(' ')