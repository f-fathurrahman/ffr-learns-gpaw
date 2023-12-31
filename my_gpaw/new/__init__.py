from collections import defaultdict
from contextlib import contextmanager
from time import time
from typing import Iterable


def prod(iterable: Iterable[int]) -> int:
    """Simple int product.

    >>> prod([])
    1
    >>> prod([2, 3])
    6
    """
    result = 1
    for x in iterable:
        result *= x
    return result


def cached_property(method):
    """Quick'n'dirty implementation of cached_property coming in Python 3.8."""
    name = f'__{method.__name__}'

    def new_method(self):
        if not hasattr(self, name):
            setattr(self, name, method(self))
        return getattr(self, name)

    return property(new_method)


def zip(*iterables, strict=True):
    """From PEP 618."""
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        pass
    if not strict:
        return
    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i+1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i+1} is longer than argument{plural}{i}"
            raise ValueError(msg)


class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.times['Total'] = -time()

    @contextmanager
    def __call__(self, name):
        t1 = time()
        try:
            yield
        finally:
            t2 = time()
            self.times[name] += t2 - t1

    def write(self, log):
        self.times['Total'] += time()
        total = self.times['Total']
        log('\ntiming:  # [seconds]')
        n = max(len(name) for name in self.times) + 2
        w = len(f'{total:.3f}')
        N = 71 - n - w
        for name, t in self.times.items():
            m = int(round(2 * N * t / total))
            bar = '━' * (m // 2) + '╸' * (m % 2)
            log(f'  {name + ":":{n}}{t:{w}.3f}  # {bar}')
