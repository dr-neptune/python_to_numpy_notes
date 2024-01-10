import random
import numpy as np

## 4.10 Introduction


def add_python(Z1, Z2):
    return [z1 + z2 for z1, z2 in zip(Z1, Z2)]

def add_numpy(Z1, Z2):
    return np.add(Z1, Z2)

vec_length = 1_000_000
Z1, Z2 = random.sample(range(vec_length), vec_length), random.sample(range(vec_length), vec_length)

# %timeit add_python(Z1, Z2)
# 253 ms ± 4.55 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit add_numpy(Z1, Z2)
# 501 ms ± 19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

def add_python(Z1, Z2):
    return [((z1**2 + z2**2)**0.5) + ((z1 + z2)**3) for z1, z2 in zip(Z1, Z2)]

def add_numpy(Z1, Z2):
    return np.sqrt(Z1**2 + Z2**2) + (Z1 + Z2)**3

vec_length = 1_000_000
Z1, Z2 = random.sample(range(vec_length), vec_length), random.sample(range(vec_length), vec_length)
Z1_np, Z2_np = np.array(Z1, dtype=np.float64), np.array(Z2, dtype=np.float64)

%timeit add_python(Z1, Z2)
# 665 ms ± 20.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit add_numpy(Z1_np, Z2_np)
# 54.2 ms ± 2.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


## 4.11 Uniform Vectorization

Z = [[0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 1, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]


def compute_neighbors(Z):
    shape = len(Z), len(Z[0])
    N = [[0,] * shape[0] for i in range(shape[1])]

    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] - 1):
            N[x][y] = Z[x-1][y-1] + Z[x][y-1] + Z[x+1][y-1] \
                + Z[x-1][y] + Z[x+1][y] \
                + Z[x-1][y+1] + Z[x][y+1] + Z[x+1][y+1]
    return N


def iterate(Z):
    N = compute_neighbors(Z)
    shape = len(Z), len(Z[0])
    for x in range(1, shape[0]-1):
        for y in range(1, shape[1]-1):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and N[x][y] == 3:
                Z[x][y] = 1
    return Z


for _ in range(10):
    Z = iterate(Z)
    print(Z)

### numpy implementation

Z = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

N = np.zeros(Z.shape, dtype=int)
N[1:-1,1:-1] += (Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
                 Z[1:-1, :-2]                + Z[1:-1,2:] +
                 Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

# flatten arrays
N_, Z_ = N.ravel(), Z.ravel()

# apply rules
R1 = np.argwhere((Z_ == 1) & (N_ < 2))
R2 = np.argwhere((Z_ == 1) & (N_ > 3))
R3 = np.argwhere((Z_ == 1) & ((N_ == 2) | (N_ == 3)))
R4 = np.argwhere((Z_ == 0) & (N_ == 3))

# set new values
Z_[R1], Z_[R2], Z_[R3], Z_[R4] = 0, 0, Z_[R3], 1

# make sure borders stay null
Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0

# numpy boolean capability
birth = (N==3)[1:-1,1:-1] & (Z[1:-1,1:-1]==0)
survive = ((N==2) | (N==3))[1:-1,1:-1] & (Z[1:-1,1:-1]==1)
Z[...] = 0
Z[1:-1,1:-1][birth | survive] = 1


## Gray-Scott
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------
n = 256
# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065  # Bacteria 1
# Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065  # Bacteria 2
# Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062  # Coral
# Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062  # Fingerprint
# Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050  # Spirals
# Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050  # Spirals Dense
# Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050  # Spirals Fast
# Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055  # Unstable
# Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065  # Worms 1
# Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063  # Worms 2
# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060  # Zebrafish


Z = np.zeros((n+2, n+2), [('U', np.double),
                          ('V', np.double)])
U, V = Z['U'], Z['V']
u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

r = 20
u[...] = 1.0
U[n//2-r:n//2+r, n//2-r:n//2+r] = 0.50
V[n//2-r:n//2+r, n//2-r:n//2+r] = 0.25
u += 0.05*np.random.uniform(-1, +1, (n, n))
v += 0.05*np.random.uniform(-1, +1, (n, n))


def update(frame):
    global U, V, u, v, im

    for i in range(10):
        Lu = (                  U[0:-2, 1:-1] +
              U[1:-1, 0:-2] - 4*U[1:-1, 1:-1] + U[1:-1, 2:] +
                                U[2:  , 1:-1])
        Lv = (                  V[0:-2, 1:-1] +
              V[1:-1, 0:-2] - 4*V[1:-1, 1:-1] + V[1:-1, 2:] +
                                V[2:  , 1:-1])
        uvv = u*v*v
        u += (Du*Lu - uvv + F*(1-u))
        v += (Dv*Lv + uvv - (F+k)*v)

    im.set_data(V)
    im.set_clim(vmin=V.min(), vmax=V.max())

fig = plt.figure(figsize=(4, 4))
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(V, interpolation='bicubic', cmap=plt.cm.viridis)
plt.xticks([]), plt.yticks([])
animation = FuncAnimation(fig, update, interval=10, frames=2000)
# animation.save('gray-scott-1.mp4', fps=40, dpi=80, bitrate=-1, codec="libx264",
#                extra_args=['-pix_fmt', 'yuv420p'],
#                metadata={'artist':'Nicolas P. Rougier'})
plt.show()



## 4.12 Temporal Vectorization

def mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    def mandelbrot(z, maxiter):
        c = z
        for n in range(maxiter):
            if abs(z) > horizon:
                return n
            z = z*z + c
        return maxiter
    r1 = [xmin + i * (xmax - xmin) / xn for i in range(xn)]
    r2 = [ymin + i * (ymax - ymin) / yn for i in range(yn)]
    return [mandelbrot(complex(r, i), maxiter) for r in r1 for i in r2]

def mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:,None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, np.complex64)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N

def mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    Xi, Yi = Xi.astype(np.uint32), Yi.astype(np.uint32)
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)[Yi]
    C = X + Y*1j
    N_ = np.zeros(C.shape, dtype=np.uint32)
    Z_ = np.zeros(C.shape, dtype=np.complex64)
    Xi.shape = Yi.shape = C.shape = xn*yn

    Z = np.zeros(C.shape, np.complex64)
    for i in range(itermax):
        if not len(Z): break

        # Compute for relevant points only
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # Failed convergence
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i+1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        I = ~I
        # np.negative(I,I)
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T

xmin, xmax, xn = -2.25, +0.75, int(3000/3)
ymin, ymax, yn = -1.25, +1.25, int(2500/3)
maxiter = 200

%timeit mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter)
%timeit mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter)
mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, maxiter)
# ~ 3.5x speedup


## Minkowski-Bouligand dimension

def fractal_dimension(Z, threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio

Z = 1.0 - imageio.imread("../from-python-to-numpy/data/Great-Britain.png")/255

print(fractal_dimension(Z, threshold=0.25))

sizes = 128, 64, 32
xmin, xmax = 0, Z.shape[1]
ymin, ymax = 0, Z.shape[0]
fig = plt.figure(figsize=(10, 5))

for i, size in enumerate(sizes):
    ax = plt.subplot(1, len(sizes), i+1, frameon=False)
    ax.imshow(1-Z, plt.cm.gray, interpolation="bicubic", vmin=0, vmax=1,
              extent=[xmin, xmax, ymin, ymax], origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for y in range(Z.shape[0]//size+1):
        for x in range(Z.shape[1]//size+1):
            s = (Z[y*size:(y+1)*size, x*size:(x+1)*size] > 0.25).sum()
            if s > 0 and s < size*size:
                rect = patches.Rectangle(
                    (x*size, Z.shape[0]-1-(y+1)*size),
                    width=size, height=size,
                    linewidth=.5, edgecolor='.25',
                    facecolor='.75', alpha=.5)
                ax.add_patch(rect)

plt.tight_layout()
# plt.savefig("fractal-dimension.png")
plt.show()

## Spatial Vectorization
