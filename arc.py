import collections
from inspect import signature
from types import FunctionType

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
    @classmethod
    def transforms(cls):
        return ({name : f for name, f in cls.__dict__.items()
                   if type(f) == FunctionType and not name.startswith('__')})

    @classmethod
    def unary_tfs(cls):
        return ({name : tf for name, tf in cls.transforms().items()
                    if len(signature(tf).parameters)==1})

    @classmethod
    def binary_tfs(cls):
        return ({name: tf for name, tf in cls.transforms().items()
                 if len(signature(tf).parameters)==2})

    def __init__(self, matrix):
        self.matrix = np.matrix(matrix)
        self.height, self.width = self.matrix.shape


    # @withrepr(lambda x: "<Func: %s>" % x.__name__)
    def rotate_90(self):
        """
        Rotate clockwise by 90 degrees.
        """
        return [Image(np.rot90(self.matrix, axes=(1, 0)))]

    def mirror(self):
        """
        Mirror flip from left to right
        [[ 1.,  2.,  3.],         [ 3.,  2.,  1.]
        [ 0.,  2.,  0.],   --->   [ 0.,  2.,  0.]
        [ 0.,  0.,  3.]]          [ 3.,  0.,  0.]
        """
        return [Image(np.fliplr(self.matrix))]

    #def repeat_2(self):
        #return Image(self.matrix.repeat(2, axis=0).repeat(2, axis=1))

    def repeat_3(self):
        """
        Copy and repeat the array in a 3x3 square area
                            [[ 1., 1., 1.],
         [[ 1.]]    --->     [ 1., 1., 1.],
                             [ 1., 1., 1.]]
        """
        return [Image(self.matrix.repeat(3, axis=0).repeat(3, axis=1))]

    # @withrepr(lambda x: x.__name__)
    def concat_right(self, image):
        """
        A,B -> AB
        """
        if self.height != image.height: return []
        return [Image(np.concatenate((self.matrix, image.matrix), axis=1))]

    # @withrepr(lambda x: x.__name__)
    def concat_down(self, image):
        """
        A,B -> A
               B
        """
        if self.width != image.width: return []
        return [Image(np.concatenate((self.matrix, image.matrix)))]

    def logical_and(self, image):
        """
        A & B
        """
        w, h = self.width, self.height
        i_w, i_h = image.width, image.height

        w, h = self.width, self.height
        i_w, i_h = image.width, image.height

        if w > i_w or h > i_h:  # try to adjust the image size
            ratio_w = w / float(i_w)
            ratio_h = h / float(i_h)
            if ratio_w.is_integer() and ratio_h.is_integer():
                image.matrix = np.tile(image.matrix, (int(ratio_h), int(ratio_w)))
            else:
                return []
        elif w != i_w or h != i_h:
            return []

        return [Image(self.matrix & image.matrix)]

    def __eq__(self, other):
        if isinstance(other, Image):
            return np.array_equal(self.matrix,other.matrix)
        return False

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)



def partitionfunc(n, k, l=1):
    """n is the integer to partition, k is the length of partitions, l is the min partition element size"""
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l, n + 1):
        for result in partitionfunc(n - i, k - 1, i):
            yield (i,) + result


def permutatations(partitions):
    return [*itertools.chain.from_iterable(set(itertools.permutations(p)) for p in partitions)]


def recursionTree(depth):
    input_tree = tree()
    input_tree['image']
    expressions = collections.defaultdict(list)
    expressions[1] = [input_tree]
    for i in range(2, depth + 1):
        # single argument
        unary_expressions = collections.defaultdict(list)
        for img_tree in expressions[i - 1]:
            for tf_name in Image.unary_tfs().keys():
                # create tree
                tf_tree = tree()
                tf_tree[tf_name] = img_tree
                # append
                unary_expressions[tf_name].append(tf_tree)
        # two arguments
        partitions = list(partitionfunc(i - 1, 2))
        perms = permutatations(partitions)
        binary_expressions = collections.defaultdict(list)
        for a, b in perms:
            for img1_tree in expressions[a]:
                for img2_tree in expressions[b]:
                    for tf_name in Image.binary_tfs().keys():
                        # create tree
                        tf_tree = tree()
                        tf_tree[tf_name]['l'] = img1_tree
                        tf_tree[tf_name]['r'] = img2_tree
                        # append
                        binary_expressions[tf_name].append(tf_tree)
        # append
        for u_expr in unary_expressions.values(): expressions[i] += u_expr
        for b_expr in binary_expressions.values(): expressions[i] += b_expr

    return expressions


def compute(exprTree, inpImage):
    if not exprTree:
        return []
    key = next(iter(exprTree.keys()))
    if key == 'image':
        return [inpImage]
    elif key in Image.binary_tfs():
        l_list = compute(exprTree[key]['l'], inpImage)
        r_list = compute(exprTree[key]['r'], inpImage)
        res = []
        if not l_list or not r_list: return res
        for l in l_list:
            for r in r_list:
                res+=Image.binary_tfs()[key](l, r)
        return res
    elif key in Image.unary_tfs():
        l_list = compute(next(iter(exprTree.values())), inpImage)
        res = []
        for l in l_list:
            res+=Image.unary_tfs()[key](l)
        return res


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
                preds = compute(imgTree, inp)
                if out not in preds:
                    passed = False
                    break
            if passed:
                solutions.append(imgTree)
                if verbose:
                    print('Bingo!')
                    pp.pprint(dicts(imgTree))

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
