# Problem 8

A square plate [-1, 1] x [-1, 1] is at temperature u = 0. At time t = 0 the
temperature is increased to u = 5 along one of the four sides while being held
at u = 0 along the other three sides, and heat then flows into the plate
according to u(t) = du. When does the temperature reach u = 1 at the center of
the plate?

## Solution

To forty digits the correct solution is

```
0.4240113870336883637974336685932564512477
```

The solution is known to more than [10,002 digits](http://www-m3.ma.tum.de/m3old/bornemann/challengebook/Chapter8/sol8_10002.txt)

## My Implementation

The solution is achieved by way of implementing a Crank-Nicholson, central
difference finite difference scheme on a 2-dimensional square of the above
dimensions. The scheme is implemented _via_ a combination of the techniques
listed here:
* Linear Operators (OpenCL kernels) to apply the numerical scheme
* The Conjugate Gradient method to apply the inverse of a matrix
* A Preconditioner matrix formed through Sparse Incomplete LU decomposition

I calculate 2 digits in 250 seconds.

_Note_: An analytical solution for this Problem exists, however I had already
built this solution as part of my MSc Scientific Computing course. It is a good
use case for OpenCL, so I thought it would include it again here


## Usage

* Install the necessary libraries listed in `requirements.txt`
* Check that OpenCL runs as expected with `python test.py`
* Solve the system with `python main.py`.

## Theory

### Overview

This project numerically solves the following problem:

> _A square plate [−1, 1] × [−1, 1] is at temperature u = 0. At time $t$ = 0
the temperature is increased to $u = 5$ along one of the four sides while
being held at $u = 0$ along the other three sides, and heat then flows into
the plate according to_ $u_t = ∆u$.
>
>_When does the temperature reach $u = 1$ at the center of the plate?_
>
> In the Moment of Heat, **SIAM 100-Digit Challenge**

The solution is achieved by way of implementing a Crank-Nicholson, central difference
finite difference scheme on a 2-dimensional square of the above dimensions. The
scheme is implemented _via_ a combination of the techniques listed here:
* Linear Operators (OpenCL kernels) to apply the numerical scheme
* The Conjugate Gradient method to apply the inverse of a matrix
* A Preconditioner matrix formed through Sparse Incomplete LU decomposition
* Numba-accelerated Python for functions that do simple calculations
* The Secant Method for finding the solution to the problem.


### Mathematical Derivation of Numerical Scheme

Where $u_h$ is the numerical approximation of the solution $u$, $\tau$ is a small time step, $h$ is a small spatial step in either of the two spatial dimensions,  the scheme can be expressed mathematically as the following:

```
u_h(t, x, y) = u_h(t - \tau, x, y) -  \frac{\tau}{2h^2}
    \left[
        4 u_h(t - \tau, x, y)
        - u_h(t - \tau, x - h, y)
        - u_h(t - \tau, x + h, y)
        - u_h(t - \tau, x, y - h)
        - u_h(t - \tau, x, y + h)
      + 4 u_h(t, x, y)
        - u_h(t, x - h, y)
        - u_h(t, x + h, y)
        - u_h(t, x, y - h)
        - u_h(t, x, y + h)
    \right]
```

Using $\alpha := \frac{\tau}{2h^2}$ and matching similar terms we arrive at the following:

```
(1 + 4\alpha)u_h(t, x, y) - \alpha[
  u_h(t, x - h, y)
  + u_h(t, x + h, y)
  + u_h(t, x, y - h)
  +  u_h(t, x, y + h)
] = (1 - 4\alpha)u_h(t - \tau, x, y) + \alpha[
  u_h(t - \tau, x - h, y)
  + u_h(t - \tau, x + h, y)
  + u_h(t - \tau, x, y - h)
  +  u_h(t - \tau, x, y + h)
]
```

Flattening the lattice row-wise and applying the boundary conditions $b$ gives the following linear system:

$$
A_{t+1}u_{h, n+1} = A_t u_{h, n} + 2\alpha b
$$

### Organisation of Algorithm
The algorithm for finding the solution follows this general algorithm:
1. Construct the Linear Operators $A_t$ and $A_{t+1}$ as OpenCL kernels
2. Construct the preconditioner $P$ to approximate the inverse of $A_{t+1}$
3. Simulate the system checking the temperature at the center of the plate after
   every iteration. Once the temperature passes the target, it returns this and
   the previous values (lower and higher bounds).
