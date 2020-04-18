from types import FunctionType
# from inspect import signature

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


def Tree(): return defaultdict(Tree)


def dicts(t):
    k = get_key(t)
    if k == 'image':
        return {k}
    else:
        return {k: dicts(t[k])}


COLORS = range(10)
OBJECT_INDICES = range(10)

class A:
    def __init__(self, matrix):
        self.matrix = matrix

def to_tuple(array):
    """
    array - list of lists of 2-dim np.array
    """
    return tuple(map(tuple, array))

class Image:
    """
    If a transformation can not be applied to given parameters, it should return None.
    Transformations have arguments (another images) and parameters (i.e., color).
    """

    @classmethod
    def transforms(cls):
        return ({name: f for name, f in cls.__dict__.items()
                 if type(f) == FunctionType and not name.startswith('__')})

    @classmethod
    def param_tfs(cls):
        return {'crop': COLORS, 'get_object': OBJECT_INDICES}

    @classmethod
    def unary_tfs(cls):
        return {'rotate_90': Image.rotate_90, 'mirror': Image.mirror, 'crop': Image.crop, 'get_object': Image.get_object}
        # return ({name : tf for name, tf in cls.transforms().items()
        #             if len(signature(tf).parameters)==1})

    @classmethod
    def binary_tfs(cls):
        return {'concat_right': Image.concat_right}#, 'concat_down': Image.concat_down, 'logical_and': Image.logical_and}
        # return ({name: tf for name, tf in cls.transforms().items()
        #          if len(signature(tf).parameters)==2})

    def __init__(self, matrix):
        """
        All fields are immutable so we could hash Image instances.
        matrix: a list of lists or numpy array
        """
        self.matrix = to_tuple(matrix)
        self.height, self.width = len(self.matrix), len(self.matrix[0])

    # @withrepr(lambda x: "<Func: %s>" % x.__name__)
    def rotate_90(self):
        """
        Rotate clockwise by 90 degrees.
        This transformation has propriety rotate_90^4(I)=I.
        This transformation commutes with mirror, crop.
        """
        matrix = np.array(self.matrix)
        return Image(np.rot90(matrix, axes=(1, 0)))

    def mirror(self):
        """
        Mirror flip from left to right
        [[1,  2,  3],         [[3,  2,  1]
        [0,  2,  0],   --->   [0,  2,  0]
        [0,  0,  3]]          [3,  0,  0]]
        This transformation has propriety mirror(mirror(I)) = I.
        This transformation commutes with rotate_90, crop.
        """
        matrix = np.array(self.matrix)
        return Image(np.fliplr(matrix))

    # @withrepr(lambda x: x.__name__)
    def concat_right(self, image):
        """
        image - argument
        A,B -> AB
        """
        if self.height != image.height:
            return None
        matrix = np.array(self.matrix)
        image_matrix = np.array(image.matrix)
        return Image(np.concatenate((matrix, image_matrix), axis=1))

    # @withrepr(lambda x: x.__name__)
    def concat_down(self, image):
        """
        image - argument
        A,B -> A
               B
        """
        if self.width != image.width:
            return None
        matrix = np.array(self.matrix)
        image_matrix = np.array(image.matrix)
        return Image(np.concatenate((matrix, image_matrix)))

    def crop(self, color):
        """
        color - parameter
        crops the minimal bound rectangle for a given color
        [[1,  2,  3],          [[3]
        [0,  2,  0],  3 --->   [0]
        [0,  0,  3]]           [3]]
        This transformation has propriety crop(crop(I, color)) = crop(I, color).
        This transformation commutes with mirror, rotate_90.
        """
        left = 100
        right = -1
        up = 100
        down = -1
        matrix = np.array(self.matrix)
        for i in range(self.height):
            for j in range(self.width):
                if matrix[i, j] == color:
                    if j < left:
                        left = j
                    if j > right:
                        right = j
                    if i < up:
                        up = i
                    if i > down:
                        down = i
        crop_matrix = matrix[up:down + 1, left:right + 1]
        if crop_matrix.size == 0:
            return None
        return Image(crop_matrix)

    # def logical_and(self, image):
    #     """
    #     A & B
    #     color + color = color
    #     color + another_color = black
    #     """
    #     w, h = self.width, self.height
    #     i_w, i_h = image.width, image.height
    #
    #     if w > i_w or h > i_h:  # try to adjust the image size
    #         ratio_w = w / float(i_w)
    #         ratio_h = h / float(i_h)
    #         if ratio_w.is_integer() and ratio_h.is_integer():
    #             image.matrix = np.tile(image.matrix, (int(ratio_h), int(ratio_w)))
    #         else:
    #             return None
    #     elif w != i_w or h != i_h:
    #         return None
    #
    #     return Image(self.matrix & image.matrix)

    def recolor(self, color1, color2):
        """
        changes color1 in the image to color2
        """
        matrix = np.array(self.matrix)
        return Image(np.where(matrix == color1, color2, matrix))

    def get_object(self, index):
        """
        Returns an element of the list of minimal rectangles for all objects in the image with index.
        Object is a connected component of more than one cell with pixels of the same color except black.
        Looks at vertical, horizontal and diagonal connections.
        This transformation has property ∀ index1 ∃ index2:
        get_object(get_object(I, index1), index2) = get_object(I, index1).
        """
        matrix = np.array(self.matrix)
        assert len(matrix.shape) == 2
        h, w = matrix.shape
        dyy = [0, -1, -1, -1, 0, 1, 1, 1]
        dxx = [1, 1, 0, -1, -1, -1, 0, 1]
        visited = np.zeros(matrix.shape, dtype=bool)
        shapes = []
        based_geometry = {'l': w, 'r': 0, 'b': h, 't': 0}
        for i in range(h):
            for j in range(w):
                color = matrix[i,j]
                if color == 0:
                    continue
                stack = [(i, j)]
                geometry = dict(based_geometry)
                while stack:
                    y, x = stack.pop(0)
                    if not (0 <= y < h and 0 <= x < w and matrix[y, x] == color):
                        continue
                    if not visited[y][x]:
                        geometry['l'] = min(geometry['l'], x)
                        geometry['r'] = max(geometry['r'], x)
                        geometry['b'] = min(geometry['b'], y)
                        geometry['t'] = max(geometry['t'], y)
                        visited[y][x] = True
                        for dy, dx in zip(dyy, dxx):
                            y_, x_ = y + dy, x + dx
                            stack.append((y_, x_))
                shared_items = {k: geometry[k] for k in geometry if
                                k in based_geometry and geometry[k] == based_geometry[k]}
                if len(shared_items) < 4:
                    crop_matrix = matrix[geometry['b']:geometry['t'] + 1, geometry['l']:geometry['r'] + 1]
                    if crop_matrix.size > 1 and crop_matrix.size != matrix.size:
                        shapes.append(Image(crop_matrix))
        if index in range(len(shapes)):
            return shapes[index]
        return None

    def __eq__(self, other):
        if isinstance(other, Image):
            return np.array_equal(np.array(self.matrix), np.array(other.matrix))
        return False

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def __hash__(self):
        return hash(self.matrix)


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


def permutations(partitions):
    return [*itertools.chain.from_iterable(set(itertools.permutations(p)) for p in partitions)]


def create_random_string(k=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


sorted_unary = sorted(Image.unary_tfs())


def recursionTree(length):
    input_tree = Tree()
    input_tree['image']
    expressions = defaultdict(list)
    expressions[1] = [input_tree]
    for i in range(2, length + 1):
        # single argument
        unary_expressions = defaultdict(list)
        for img_tree in expressions[i - 1]:
            for tf_name in sorted_unary:
                prev_tf_name = get_key(img_tree)
                # if previous transformation is lexicographically smaller, don't apply this, because they commute
                # i.e. no need to append b(a(...)) because we've already appended a(b(...))
                # check prev_tf_name >= tf_name for commuting or prev not in commuting (which == sorted_unary here)
                # get_object does not commute with others but for simplicity we assume that too
                if prev_tf_name >= tf_name or prev_tf_name not in sorted_unary:
                    # check if previous transformation is not crop because crop^2(I) = crop(I)
                    # this is true only for the same color but we suppose that we can't crop^2 anyways
                    if tf_name == 'crop':
                        tf_tree = None
                        if prev_tf_name != 'crop':
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    # check if previous transformation is not mirror because mirror^2(I) = I
                    elif tf_name == 'mirror':
                        tf_tree = None
                        if prev_tf_name != 'mirror':
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    elif tf_name == 'get_object':
                        tf_tree = None
                        if prev_tf_name != 'get_object':
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    else:
                        # create tree
                        tf_tree = Tree()
                        tf_tree[tf_name] = img_tree
                    # append
                    # unary_expressions[tf_name] += [tf_tree] if tf_tree else []
                    if tf_tree:
                        unary_expressions[tf_name].append(tf_tree)
        # two arguments
        partitions = list(partitionfunc(i - 1, 2))
        perms = permutations(partitions)
        binary_expressions = defaultdict(list)
        for a, b in perms:
            for img1_tree in expressions[a]:
                for img2_tree in expressions[b]:
                    for tf_name in Image.binary_tfs().keys():
                        # don't add concat(mirror,mirror) because mirror(concat(,))
                        if tf_name in ['concat_right', 'concat_down']:
                            tf_tree = None
                            if not (get_key(img1_tree) == 'mirror' and
                                    get_key(img2_tree) == 'mirror'):
                                tf_tree = Tree()
                                tf_tree[tf_name]['l'] = img1_tree
                                tf_tree[tf_name]['r'] = img2_tree
                        else:
                            # create tree
                            tf_tree = Tree()
                            tf_tree[tf_name]['l'] = img1_tree
                            tf_tree[tf_name]['r'] = img2_tree
                        # append
                        if tf_tree:
                            binary_expressions[tf_name].append(tf_tree)
        # append
        for u_expr in unary_expressions.values(): expressions[i] += u_expr
        for b_expr in binary_expressions.values(): expressions[i] += b_expr

    return expressions

def recursionTreeEvalSingle(length, image):
    input_tree = Tree()
    input_tree['image']
    expressions = defaultdict(list)
    expressions[1] = [(input_tree,[image])]
    for i in range(2, length + 1):
        # single argument
        unary_expressions = defaultdict(list)
        for img_tree, prev_list_of_imgs in expressions[i - 1]:
            for tf_name in sorted_unary:
                prev_tf_name = get_key(img_tree)
                # if previous transformation is lexicographically smaller, don't apply this, because they commute
                # i.e. no need to append b(a(...)) because we've already appended a(b(...))
                # check prev_tf_name >= tf_name for commuting or prev not in commuting (which == sorted_unary here)
                # get_object does not commute with others but for simplicity we assume that too
                if prev_tf_name >= tf_name or prev_tf_name not in sorted_unary:
                    # check if previous transformation is not crop because crop^2(I) = crop(I)
                    # this is true only for the same color but we suppose that we can't crop^2 anyways
                    if tf_name == 'crop':
                        tf_tree = None
                        if prev_tf_name != 'crop':
                            list_of_imgs = []
                            for img in prev_list_of_imgs:
                                for param in Image.param_tfs()[tf_name]:
                                    v = Image.unary_tfs()[tf_name](img, param)
                                    if v:
                                        list_of_imgs.append(v)
                            if list_of_imgs:
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    # check if previous transformation is not mirror because mirror^2(I) = I
                    elif tf_name == 'mirror':
                        tf_tree = None
                        if prev_tf_name != 'mirror':
                            list_of_imgs = []
                            for img in prev_list_of_imgs:
                                v = Image.unary_tfs()[tf_name](img)
                                if v:
                                    list_of_imgs.append(v)
                            if list_of_imgs:
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    # check if previous transformation is not crop because get_object^2(I) = get_object(I)
                    # this is true only for the same index but we suppose that we can't get_object^2 anyways
                    elif tf_name == 'get_object':
                        tf_tree = None
                        if prev_tf_name != 'get_object':
                            list_of_imgs = []
                            for img in prev_list_of_imgs:
                                for param in Image.param_tfs()[tf_name]:
                                    v = Image.unary_tfs()[tf_name](img, param)
                                    if v:
                                        list_of_imgs.append(v)
                            # create tree
                            if list_of_imgs:
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    else:
                        tf_tree = None
                        list_of_imgs = []
                        for img in prev_list_of_imgs:
                            v = Image.unary_tfs()[tf_name](img)
                            if v:
                                list_of_imgs.append(v)
                        # create tree
                        if list_of_imgs:
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    # append
                    # unary_expressions[tf_name] += [tf_tree] if tf_tree else []
                    if tf_tree:
                        unary_expressions[tf_name].append((tf_tree, list_of_imgs))
        # append
        for u_expr in unary_expressions.values():
            expressions[i] += u_expr

    return expressions

def recursionTreeEvalMany(length, images):
    input_tree = Tree()
    input_tree['image']
    expressions = defaultdict(list)
    expressions[1] = [(input_tree,images)]
    for i in range(2, length + 1):
        # single argument
        unary_expressions = defaultdict(list)
        for img_tree, prev_list_of_list_of_imgs in expressions[i - 1]:
            for tf_name in sorted_unary:
                prev_tf_name = get_key(img_tree)
                # if previous transformation is lexicographically smaller, don't apply this, because they commute
                # i.e. no need to append b(a(...)) because we've already appended a(b(...))
                # check prev_tf_name >= tf_name for commuting or prev not in commuting (which == sorted_unary here)
                # get_object does not commute with others but for simplicity we assume that too
                if prev_tf_name >= tf_name or prev_tf_name not in sorted_unary:
                    # check if previous transformation is not crop or get_object because tf^2(I) = tf(I)
                    # this is true only for the same param but we suppose that we can't tf^2 anyways
                    if tf_name in ['crop', 'get_object']:
                        tf_tree = None
                        if prev_tf_name != tf_name:
                            list_of_list_of_imgs = []
                            for prev_list_of_imgs in prev_list_of_list_of_imgs:
                                list_of_imgs = []
                                for img in prev_list_of_imgs:
                                    for param in Image.param_tfs()[tf_name]:
                                        v = Image.unary_tfs()[tf_name](img, param)
                                        if v:
                                            list_of_imgs.append(v)
                                list_of_list_of_imgs.append(list_of_imgs)
                            # check that is every element of list_of_list_of_imgs is not empty
                            if all(list_of_list_of_imgs):
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    # check if previous transformation is not mirror because mirror^2(I) = I
                    elif tf_name == 'mirror':
                        tf_tree = None
                        if prev_tf_name != tf_name:
                            list_of_list_of_imgs = []
                            for prev_list_of_imgs in prev_list_of_list_of_imgs:
                                list_of_imgs = []
                                for img in prev_list_of_imgs:
                                    v = Image.unary_tfs()[tf_name](img)
                                    if v:
                                        list_of_imgs.append(v)
                                list_of_list_of_imgs.append(list_of_imgs)
                            if all(list_of_list_of_imgs):
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    else:
                        tf_tree = None
                        list_of_list_of_imgs = []
                        for prev_list_of_imgs in prev_list_of_list_of_imgs:
                            list_of_imgs = []
                            for img in prev_list_of_imgs:
                                v = Image.unary_tfs()[tf_name](img)
                                if v:
                                    list_of_imgs.append(v)
                            list_of_list_of_imgs.append(list_of_imgs)
                        # create tree
                        if all(list_of_list_of_imgs):
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    # append
                    # unary_expressions[tf_name] += [tf_tree] if tf_tree else []
                    if tf_tree:
                        unary_expressions[tf_name].append((tf_tree, list_of_list_of_imgs))
        # append
        for u_expr in unary_expressions.values():
            expressions[i] += u_expr

    return expressions


def recursionTreeEvalSet(length, images):
    list_of_sets_of_all_possible_images = [set() for _ in range(len(images))]
    input_tree = Tree()
    input_tree['image']
    expressions = defaultdict(list)
    expressions[1] = [(input_tree,images)]
    for i in range(2, length + 1):
        # single argument
        unary_expressions = defaultdict(list)
        for img_tree, prev_list_of_list_of_imgs in expressions[i - 1]:
            for tf_name in sorted_unary:
                prev_tf_name = get_key(img_tree)
                # if previous transformation is lexicographically smaller, don't apply this, because they commute
                # i.e. no need to append b(a(...)) because we've already appended a(b(...))
                # check prev_tf_name >= tf_name for commuting or prev not in commuting (which == sorted_unary here)
                # get_object does not commute with others but for simplicity we assume that too
                if prev_tf_name >= tf_name or prev_tf_name not in sorted_unary:
                    # check if previous transformation is not crop or get_object because tf^2(I) = tf(I)
                    # this is true only for the same param but we suppose that we can't tf^2 anyways
                    if tf_name in ['crop', 'get_object']:
                        tf_tree = None
                        if prev_tf_name != tf_name:
                            list_of_list_of_imgs = []
                            for ind, prev_list_of_imgs in enumerate(prev_list_of_list_of_imgs):
                                list_of_imgs = set()  # should be set !
                                for img in prev_list_of_imgs:
                                    for param in Image.param_tfs()[tf_name]:
                                        v = Image.unary_tfs()[tf_name](img, param)
                                        if v and v not in list_of_sets_of_all_possible_images[ind]:
                                            list_of_imgs.add(v)
                                            list_of_sets_of_all_possible_images[ind].add(v)
                                list_of_list_of_imgs.append(list_of_imgs)
                            # check that is every element of list_of_list_of_imgs is not empty
                            if all(list_of_list_of_imgs):
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    # check if previous transformation is not mirror because mirror^2(I) = I
                    elif tf_name == 'mirror':
                        tf_tree = None
                        if prev_tf_name != tf_name:
                            list_of_list_of_imgs = []
                            for ind, prev_list_of_imgs in enumerate(prev_list_of_list_of_imgs):
                                list_of_imgs = set()  # should be set !
                                for img in prev_list_of_imgs:
                                    v = Image.unary_tfs()[tf_name](img)
                                    if v and v not in list_of_sets_of_all_possible_images[ind]:
                                        list_of_imgs.add(v)
                                        list_of_sets_of_all_possible_images[ind].add(v)
                                list_of_list_of_imgs.append(list_of_imgs)
                            if all(list_of_list_of_imgs):
                                tf_tree = Tree()
                                tf_tree[tf_name] = img_tree
                    else:
                        tf_tree = None
                        list_of_list_of_imgs = []
                        for ind, prev_list_of_imgs in enumerate(prev_list_of_list_of_imgs):
                            list_of_imgs = set()  # should be set !
                            for img in prev_list_of_imgs:
                                v = Image.unary_tfs()[tf_name](img)
                                if v and v not in list_of_sets_of_all_possible_images[ind]:
                                    list_of_imgs.add(v)
                                    list_of_sets_of_all_possible_images[ind].add(v)
                            list_of_list_of_imgs.append(list_of_imgs)
                        # create tree
                        if all(list_of_list_of_imgs):
                            tf_tree = Tree()
                            tf_tree[tf_name] = img_tree
                    # append
                    # unary_expressions[tf_name] += [tf_tree] if tf_tree else []
                    if tf_tree:
                        unary_expressions[tf_name].append((tf_tree, list_of_list_of_imgs))
        # append
        for u_expr in unary_expressions.values():
            expressions[i] += u_expr

    return expressions, list_of_sets_of_all_possible_images


def solve_task(path, length, verbose=False):
    data = get_data(path)
    train_data = data['train']
    expressions, list_of_sets_of_all_possible_images = \
        recursionTreeEvalSet(length,[{Image(pair['input'])} for pair in train_data])

    # Run this if just want to know if solved or not
    # passed = True
    # for i, pair in enumerate(train_data):
    #     if Image(pair['output']) not in list_of_sets_of_all_possible_images[i]:
    #         passed = False
    #         break
    # if passed and verbose:
    #     print('Bingo!', path)

    solutions = []
    for length_of_expression, list_of_pairs_expression_list in expressions.items():
        for expression, list_of_sets in list_of_pairs_expression_list:
            passed = True
            for i, pair in enumerate(train_data):
                if Image(pair['output']) not in list_of_sets[i]:
                    passed = False
                    break
            if passed and verbose:
                print('Bingo! Length', length_of_expression, path)
                pp.pprint(dicts(expression))
            if passed:
                if verbose:
                    print('Bingo!')
                    pp.pprint(dicts(expression))
                solutions.append(expression)
    return solutions


def get_key(t):
    """
    t - tree
    """
    return next(iter(t.keys()))


def compute(expr_tree, inp_image):
    if not expr_tree:
        return []
    key = get_key(expr_tree)
    if key == 'image':
        return [inp_image]
    elif key in Image.unary_tfs():
        imgs = compute(next(iter(expr_tree.values())), inp_image)
        computes = []
        if key in Image.param_tfs():
            for param in Image.param_tfs()[key]:
                for img in imgs:
                    if img:
                        v = Image.unary_tfs()[key](img, param)
                        if v:
                            computes.append(v)
        else:
            for img in imgs:
                if img:
                    v = Image.unary_tfs()[key](img)
                    if v:
                        computes.append(v)
        return computes
    elif key in Image.binary_tfs():
        l_list = compute(expr_tree[key]['l'], inp_image)
        r_list = compute(expr_tree[key]['r'], inp_image)
        computes = []
        # if not l_list or not r_list: return computes
        for l in l_list:
            for r in r_list:
                if l and r:
                    v = Image.binary_tfs()[key](l, r)
                    if v:
                        computes.append(v)
        return computes


data_path = Path('data/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'


def get_data(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task


def find_tree(path, expressions, verbose=False):
    data = get_data(path)
    train_data = data['train']
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


def solve_all_tasks(length, training):
    expressions = recursionTree(length)
    all_solutions = {}
    path = data_path / training
    tasks = sorted(os.listdir(path))
    for task in tasks:
        task_solutions = find_tree(path / task, expressions)
        if task_solutions:
            all_solutions[task] = task_solutions
    return all_solutions


if __name__ == '__main__':
    tasks = sorted(os.listdir('data/training'))
    for task in tasks:
        solve_task('data/training/' + task, 4, verbose=True)

    # n = 4
    # inp1 = Image([[1, 1, 1],
    #             [0, 0, 0],
    #             [0, 0, 0]])
    # inp2 = Image([[1, 2, 1],
    #            [0, 1, 3],
    #            [1, 0, 1]])
    # expr = recursionTreeEvalSet(n, [{inp1},{inp2}])
    # # expressions = recursionTree(n)
    # for j in range(1, n + 1):
    #     print(j)
    #     for t, imgs1 in expr[j]:
    #         pp.pprint(dicts(t))
    #         print(imgs1)
        # pp.pprint(list(map(dicts, expressions[i])))

    # dataset = 'evaluation'
    # print(n, dataset)
    # all_solutions = solve_all_tasks(n, dataset)
    # print('Solved', len(all_solutions))
    # print()
    # for key, value in all_solutions.items():
    #     print(key)
    #     pp.pprint(dicts(value[0]))
    #     print()
