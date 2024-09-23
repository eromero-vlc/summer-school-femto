# Intro to scientific computing

Your computer is the pinnacle of human tools!
Master it or other will master you!

Programming languages allow you to harvest their power!

## Prep

We will need a BLAS and MPI libraries. Possible ways for C/C++:

1. Install OpenBLAS and Open-MPI from repositories (rpm, apt, brew)

2. Install Intel compilers, (not support for MacOS, what a surprise!)
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
   Install also Intel MPI. They have docker container.

For Python 3.7+:

1. (Optional, but encouraged) Create a new environment:

```bash
python -mvenv py3  # create a new dir with that name
. py3/bin/activate # activate the environment
```

2. Install `numba` and `mpi4py`

```bash
pip install numba
pip install mpi4py-mpich
```

## Collatz conjecture

Exercise: check that Collatz conjecture holds for all numbers smaller than 9999999.
As a check, return the total number of iterations in module 4.

Example in C:
```C
int main() {
  unsigned int s = 0;
  for (int i = 2; i < 9999999; ++i) {
    unsigned int n = i;
    while (n != 1) {
      n = n % 2 == 0 ? n / 2 : 3 * n + 1;
      ++s;
    }
  }
  return s % 4;
}
```

Run:
```bash
gcc -O3 collatz.c -o collatz
time ./collatz
# 
# real    0m2.886s
# user    0m2.878s
# sys     0m0.004s
```

Example in Python:
```Python
import sys

def f():
  s = 0
  for i in range(2,9999999):
      n = i 
      while n != 1:
          if n % 2 == 0: 
              n = n / 2     
          else:   
              n = 3 * n + 1   
          s += 1
  return s % 4

sys.exit(f())
```

Run:
```bash
time python collatz.py
# 
# real    3m55.438s
# user    3m55.379s
# sys     0m0.008s
```

Example in Python with numba:
```Python
from numba import njit
import sys

@njit
def f():
  s = 0
  for i in range(2,9999999):
      n = i
      while n != 1:
          if n % 2 == 0:
              n = n / 2
          else:
              n = 3 * n + 1
          s += 1
  return s % 4

sys.exit(f())
```

Run:
```bash
time python collatz-numba.py
# 
# real    0m28.983s
# user    0m29.314s
# sys     0m0.121s
```

### Message Passing Interface

```Python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

print("I am process {} in {} processes\n".format(rank, nproc))
```

Run:
```bash
mpiexec -np 2 python hello-mpi.py
#
# I am process 1 in 2 processes
# I am process 0 in 2 processes
```

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

print("I am process {} in {} processes\n".format(rank, nproc))
```

Example of how to sum arrays up among all processes:
```Python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# The value of `v` is different in each process
v = rank * 2

# Sum up the value of `v` among all processes
array_in = np.zeros(1, 'd')
array_sum = np.zeros(1, 'd')
array_in[0] = v
comm.Allreduce([array_in, MPI.DOUBLE], [array_sum, MPI.DOUBLE])

print("I am process {}, v={}, and s={}\n".format(rank, v, array_sum[0]))
```

#### Exercise:

Parallelize the `collatz.py` example. All processes should have the final number of steps (variable `s`) on module 4.

## Work with sparse matrices

Sparse matrices efficiently represent matrices with a few nonzeros. They are created by giving the row index, the column index and the value for each nonzero.
Example:

```Python
from scipy.sparse import csr_matrix

data = [1, 2, 3, 4]
row_indices = [0, 0, 1, 2]
col_indices = [0, 1, 2, 3]
csr_matrix_example = csr_matrix((data, (row_indices, col_indices)), shape=(3, 3))
dense_matrix = csr_matrix_example.toarray()
```

### Exercise

Generate a 1D Laplacian matrix given the size in sparse format:

```Python
def lap(n):
   ...


lap(5).toarray()
array([[ 2, -1,  0,  0,  0],
       [-1,  2, -1,  0,  0],
       [ 0, -1,  2, -1,  0],
       [ 0,  0, -1,  2, -1],
       [ 0,  0,  0, -1,  2]])
```

Time the multiplication of `lap(400)` (sparse) and `lap(400).toarray()` (dense) against a matrix of shape `(400,k)` varying `k` = 1, 2, 4, 8, ...

```Python
import timeit
Asp = lap(400)
Ade = Asp.toarray()

print("k time_sparse time_dense")
for k in [1, 2, 4, 8, 16]:
  x = np.ones(400, k)
  tsp = timeit.timeit(lambda: Asp.dot(x), number=10)
  tde = timeit.timeit(lambda: Ade.dot(x), number=10)
  print("{} {} {}".format(k, tsp, tde))
```

### Exercise

Solve a liner system with `lap(400)` using any of the linear solvers in `scipy.sparse`. For instance:

```Python
from scipy.sparse.linalg import bicgstab

A = lap(400)
b = np.ones(400,1)

# solve the liner system with accuracy |A*x - b|_2 <= 1e-5 * |b|_2
x, exit_code = bicgstab(A, b, atol=1e-5)
```

Time the difference between solving a linear system with `bicgstab` and computing the inverse with `numpy.linalg.inv`
