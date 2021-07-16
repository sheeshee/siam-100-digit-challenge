# Problem 1

What is the value of the following?

![equation](eqn.svg)

## Solution

To forty digits the correct solution is

```
0.3233674316777787613993700879521704466510
```

The solution is known to more than [10,002 digits](http://www-m3.ma.tum.de/m3old/bornemann/challengebook/Chapter1/sol1_10002.txt).

## My Implementation

I use NumPy, SciPy and Numba to implement in a Jupyter Notebook the numerical
scheme described by Walter Gautschi in his short paper
["The numerical evaluation of a challenging integral"](gautschi.pdf).

I was able to calculate 13 correct digits in about 4 ms.

## More info

Some interesting resources are linked at the bottom of this page:

http://www-m3.ma.tum.de/m3old/bornemann/challengebook/Chapter1/
