import numpy as np

vector_z = np.ones(4 * 1000000, np.float32)

vector_z[...] = 0

(np.arange(9)
 .reshape(3, 3)
 .astype(np.int16))

## direct and indirect access

# distinguish between indexing and fancy indexing
# indexing returns a view while fancy indexing returns a copy
# indexing modifies the base array

# mutation! indexing
vector_z = np.zeros(9)
z_view = vector_z[:3]
z_view[...] = 1
vector_z

# fancy indexing, without mutation
vector_z = np.zeros(9)
z_copy = vector_z[[0, 1, 2]]
z_copy[...] = 1
vector_z
z_copy

# if you need fancy indexing, it's best to keep a copy around and work with that
z_index = [0, 1, 2]
vector_z[z_index] = 1
vector_z

# we can see if our result is a view or a copy with base
# if vector.base is None, then we have a copy
vector_z = np.random.uniform(0, 1, (5, 5))

z1, z2 = vector_z[:3, :], vector_z[z_index, :]

np.allclose(z1, z2)
z1.base is vector_z
z2.base is vector_z
z2.base is None  # z2 is a acopy

# some np functions return a view when possible (e.g. ravel) and some always
# return a copy (e.g. flatten)
vector_z = np.zeros((5, 5))
vector_z.ravel().base is vector_z

vector_z[::2, ::2].ravel().base is vector_z

vector_z.flatten().base is vector_z

## temporary copy

# the most general case is the implicit creation of intermediate copies.
# this is the case with arithmetic with arrays
X, Y = np.ones(10, dtype=int), np.ones(10, dtype=int)
A = 2*X + 2*Y

# in the example above, 3 intermediate arrays were created
# 2*X, 2*Y, 2X + 2*Y

# if only the final result matters and you don't need x or y after, alternatively we can do
X, Y = np.ones(10, dtype=int), np.ones(10, dtype=int)

np.add(np.multiply(X, 2, out=X),
       np.multiply(Y, 2, out=Y))

# in the above, no temporary arrays are created

## 3.4 Conclusion
Z1 = np.arange(10)
Z2 = Z1[1:-1:2]

Z2.base is Z1  # => Z2 is a view of Z1, i.e. Z2 can be expressed as Z1[start:stop:step]

# now we want to find the start, stop, step

# for step, we can use strides to find the number of bytes to go from one element to another
# in each dimension
step = Z2.strides[0] // Z1.strides[0]

# now that we have step, we want to find start and stop indices
# we can use the byte_bounds method to get pointers to the end points of an array
offset_start, offset_stop = (np.byte_bounds(Z2)[0] - np.byte_bounds(Z1)[0]), (np.byte_bounds(Z2)[-1] - np.byte_bounds(Z1)[-1])

start, stop = offset_start // Z1.itemsize, Z1.size + offset_stop // Z1.itemsize

print(start, stop, step)

# test our results
np.allclose(Z1[start:stop:step], Z2)
