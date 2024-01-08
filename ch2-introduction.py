import random
import numpy as np


## 2.1 Simple Example

class RandomWalker:
    def __init__(self):
        self.position = 0

    def walk(self, n):
        self.position = 0
        for i in range(n):
            yield self.position
            self.position += 2 * random.randint(0, 1) - 1

walker = RandomWalker()
walk = [position for position in walker.walk(1000)]
# %timeit [position for position in walker.walk(1000)]
# 731 µs ± 91.4 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

## procedural approach

def random_walk(n):
    position = 0
    walk = [position]

    for i in range(n):
        position += 2 * random.randint(0, 1) - 1
        walk.append(position)

    return walk
# %timeit random_walk(1000)
# 572 µs ± 13.3 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

## vectorized approach
from itertools import accumulate

def random_walk_faster(n=1000):
    steps = random.choices([-1, +1], k=n)
    return [0] + list(accumulate(steps))

# %timeit random_walk_faster()
# 140 µs ± 8.41 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

def random_walk_fastest(n=1000):
    return np.random.choice([-1, 1], n).cumsum()

# %timeit random_walk_fastest()
# 29.5 µs ± 432 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
