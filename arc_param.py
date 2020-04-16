import numpy as np
import itertools
from collections import defaultdict
import random, string
import json
from pathlib import Path
import os
# from utils import withrepr

import pprint
pp = pprint.PrettyPrinter(indent=0)

def tree(): return defaultdict(tree)

def dicts(t): return {k: dicts(t[k]) for k in t}

class Image:

    def __init__(self, matrix):
        self.matrix = np.matrix(matrix)
        self.height, self.width = self.matrix.shape

    # @withrepr(lambda x: "<Func: %s>" % x.__name__)
    def rotate_90(self):
        """
        Rotate clockwise by 90 degrees.
        """
        return Image(np.rot90(self.matrix, axes=(1, 0)))

    def mirror(self):
        """
        Mirror flip from left to right
        [[ 1.,  2.,  3.],         [ 3.,  2.,  1.]
        [ 0.,  2.,  0.],   --->   [ 0.,  2.,  0.]
        [ 0.,  0.,  3.]]          [ 3.,  0.,  0.]
        """
        return Image(np.fliplr(self.matrix))

    # @withrepr(lambda x: x.__name__)
    def concat_right(self, image):
        """
        A,B -> AB
        """
        return Image(np.concatenate((self.matrix, image.matrix), axis=1))

    def concat_right_compat(self, image):
        return self.height == image.height

    # @withrepr(lambda x: x.__name__)
    def concat_down(self, image):
        """
        A,B -> A
               B
        """
        return Image(np.concatenate((self.matrix, image.matrix)))

    def concat_down_compat(self, image):
        return self.width == image.width

    def crop(self, color):
        """
        crops the minimal bound rectangle for a given color
        [[ 1.,  2.,  3.],          [ 3.]
        [ 0.,  2.,  0.],  3 --->   [ 0.]
        [ 0.,  0.,  3.]]           [ 3.]
        """
        left = 100
        right = -1
        up = 100
        down = -1
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i,j] == color:
                    if j < left: left = j
                    if j > right: right = j
                    if i < up: up = i
                    if i > down: down = i
        return Image(self.matrix[up:down+1,left:right+1])

    def __eq__(self, other):
        if isinstance(other, Image):
            return np.array_equal(self.matrix,other.matrix)
        return False

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

# methods_list = [func for func in dir(Image) if callable(getattr(Image, func)) and not func.startswith("__")]
# print(methods_list)
# for method in methods_list:
# 	exec(f"{method} = Image.{method}")

rotate_90 = Image.rotate_90
mirror = Image.mirror
concat_right = Image.concat_right
concat_right_compat = Image.concat_right_compat
concat_down = Image.concat_down
concat_down_compat = Image.concat_down_compat
crop = Image.crop

def partitionfunc(n, k, l=1):
    """n is the integer to partition, k is the length of partitions, l is the min partition element size"""
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l, n + 1):
        for result in partitionfunc(n - i, k - 1, i):
            yield (i,) + result


def permutatations(partitions):
    return [*itertools.chain.from_iterable(set(itertools.permutations(p)) for p in partitions)]


def recursion(depth, image):
    expressions = {1: [image]}
    for i in range(2, depth + 1):
        # single argument
        rotate_90_expressions = [rotate_90(img) for img in expressions[i - 1]]
        # two arguments
        partitions = list(partitionfunc(i - 1, 2))
        perms = permutatations(partitions)
        concat_right_expressions = []
        concat_down_expressions = []
        for a, b in perms:
            concat_right_expressions += [concat_right(img1, img2) for img1 in expressions[a]
                                             for img2 in expressions[b]
                                             if concat_right_compat(img1, img2)]
            concat_down_expressions += [concat_down(img1, img2) for img1 in expressions[a]
                                            for img2 in expressions[b]
                                            if concat_down_compat(img1, img2)]
        # append
        expressions[i] = (rotate_90_expressions + concat_right_expressions + concat_down_expressions)
    return expressions


def create_random_string(k=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


def recursionTree(depth):
    input_tree = tree()
    input_tree['image']
    expressions = {1: [input_tree]}
    for i in range(2, depth + 1):
        # single argument
        rotate_90_expressions = []
        mirror_expressions = []
        crop_expressions = []
        for img_tree in expressions[i - 1]:
            # create tree
            rotate_90_tree = tree()
            rotate_90_tree['rotate_90'] = img_tree
            mirror_tree = tree()
            mirror_tree['mirror'] = img_tree
            crop_tree = tree()
            crop_tree['crop'] = img_tree
            # append
            rotate_90_expressions.append(rotate_90_tree)
            mirror_expressions.append(mirror_tree)
            crop_expressions.append(crop_tree)
        # two arguments
        partitions = list(partitionfunc(i - 1, 2))
        perms = permutatations(partitions)
        concat_right_expressions = []
        concat_down_expressions = []
        for a, b in perms:
            for img1_tree in expressions[a]:
                for img2_tree in expressions[b]:
                    # if concat_right_compat(img1, img2):
                    if True:
                        # create tree
                        concat_right_tree = tree()
                        concat_right_tree['concat_right']['l'] = img1_tree
                        concat_right_tree['concat_right']['r'] = img2_tree
                        # append
                        concat_right_expressions.append(concat_right_tree)
                    # if concat_down_compat(img1, img2):
                    if True:
                        # create tree
                        concat_down_tree = tree()
                        concat_down_tree['concat_down']['l'] = img1_tree
                        concat_down_tree['concat_down']['r'] = img2_tree
                        # append
                        concat_down_expressions.append(concat_down_tree)
        # append
        expressions[i] = rotate_90_expressions + mirror_expressions + crop_expressions \
                         + concat_right_expressions + concat_down_expressions
    return expressions


def compute(exprTree, inpImage):
    if not exprTree:
        return []
    key = next(iter(exprTree.keys()))
    if key == 'image':
        return [inpImage]
    elif key == 'rotate_90':
        computes = []
        for img in compute(next(iter(exprTree.values())), inpImage):
            if img:
                computes.append(rotate_90(img))
        return computes
    elif key == 'mirror':
        computes = []
        for img in compute(next(iter(exprTree.values())), inpImage):
            if img:
                computes.append(mirror(img))
        return computes
    elif key == 'crop':
        colors = range(10)
        computes = []
        for color in colors:
            for img in compute(next(iter(exprTree.values())), inpImage):
                if img:
                    v = crop(img, color)
                    # put check in another function in Image class
                    if v.matrix.size != 0:
                        computes.append(v)
        return computes
    elif key == 'concat_right':
        computes = []
        for l in compute(exprTree['concat_right']['l'], inpImage):
            for r in compute(exprTree['concat_right']['r'], inpImage):
                if l and r and concat_right_compat(l, r):
                    computes.append(concat_right(l, r))
        return computes
    elif key == 'concat_down':
        computes = []
        for l in compute(exprTree['concat_down']['l'], inpImage):
            for r in compute(exprTree['concat_down']['r'], inpImage):
                if l and r and concat_down_compat(l, r):
                    computes.append(concat_down(l, r))
        return computes

data_path = Path('data/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'


def get_data(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task


def find_tree(path, depth, verbose=False):
    data = get_data(path)
    train_data = data['train']
    expressions = recursionTree(depth)
    solutions = []
    for i, v in expressions.items():
        for imgTree in v:
            passed = True
            for pair in train_data:
                inp, out = Image(pair['input']), Image(pair['output'])
                if out not in compute(imgTree, inp):
                    passed = False
                    break
            if passed:
                if verbose:
                    print('Bingo!')
                    pp.pprint(dicts(imgTree))
                solutions.append(imgTree)
    return solutions


def solve_all_tasks(depth, training):
    all_solutions = {}
    path = data_path / training
    tasks = sorted(os.listdir(path))
    for task in tasks:
        task_solutions = find_tree(path / task, depth)
        if task_solutions:
            all_solutions[task] = task_solutions
    return all_solutions



all_solutions = solve_all_tasks(3, 'e')
for key, value in all_solutions.items():
    print(key)
    print(value[0])
    print()
