# Problem 2: Reliability Amid Chaos

A photon moving at speed 1 in the x-y plane starts at time t = 0 at
(x, y) = (1/2, 1/10) heading due east. Around every integer lattice point (i, j)
in the plane, a circular mirror of radius 1/3 has been erected. How far from
(0, 0) is the photon at t = 10?

## Solution

To eighty digits the correct solution is

```
0.9952629194433541608903118094267216210294669227341543498032088580729861796228306
```

The solution is known to more than [10,002 digits](http://www-m3.ma.tum.de/m3old/bornemann/challengebook/Chapter2/sol2_10002.txt)

## My Implementation

This solution uses Julia to iterate through photon reflections until the the target
time has been surpassed, at which time it finds the point on the last leg of the
particle's journey where the elapsed time is equal to the target time.

By using Julia's BigFloat objects, this solution determines the correct answer
to 66 digits.

```
0.9952629194433541608903118094267216210294669227341543498032088580731118171262683
                                                                   ^- solution diverges here
```
