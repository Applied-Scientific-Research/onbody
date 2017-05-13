# onbody

Test C++ code for equivalent particle approximation technique and O(NlogN) and O(N) summation methods


## Summary

`onbody` is a testbed C++ program for a single evaluation of the gravitational forces
of a set of particles on a system of target points. Key goals are to learn:

* How C++ data structures and algorithms will work for this problem
* Whether a system that uses equivalent particles at all levels of a tree can generate solutions of sufficient accuracy
* Whether using a hierarchical system of equivalent target points allows creation of an O(N) fast summation method of sufficient accuracy and low programmer effort

The "equivalent particle" technique is where, given a `blockSize` representing the number of particles in all but one leaf node, every other node in the tree contains `blockSize` "equivalent particles" which are used to compute the influence of that box on target points. A simple treecode requires only the source points to have equivalent particles; and in this sense, they act like the multipole approximation of that tree node. For an O(N) method, the target tree also uses this system, so that all but one node in every level of the tree contains exactly `blockSize` equivalent target points, onto which the sources' influences are calculated, and from which those influences are passed down to its child boxes' equivalent target points. This means that every box-box, box-particle, or particle-box interaction can use the exact same computation kernel: `blockSize` (equivalent) particles affecting `blockSize` (equivalent) target points. This should simplify the programming of an O(N) fast summation code considerably, as long as appropriate trees and equivalent particles can be created efficiently.


## Build and run

    make
    ./onbody -n=10000

## Development notes and performance

All tests below run on MBP 12,1 with Intel i7 at 3.1 GHz, on MacOS 10.11. Program is single-threaded, compiled with `-O2` and uses block size of 64.
All reported times are in Mcycles (millions of clock tics).
The inner kernel is 19 flops, so N=32768 means 20.4 GFlops for the O(N^2) method.

#### 1st order treecode (boxes are singular particles)

N      |naive  |nmult | tree=3| tmult | tbuild| tbmult
-------|-------|------|-------|-------|-------|-------
64     |0.127  |      | 0.127 |       | 0.013 | 
512    |9.371  |73.79x| 4.894 | 38.18x| 0.424 | 32.62x
4096   |625.77 |66.78x| 167.29| 34.50x| 5.103 | 12.03x
32768  |48637. |77.72x| 2573.1| 15.38x| 113.76| 22.29x
262144 |35625+2|73.25x| 30103.| 11.70x| 7391.2| 64.97x
2097152|na     |na    | 313147| 10.40x| 643415| 87.05x

The big run took 138MB. Tree build is O(N^2) or worse - why? was std:copy on full arrays!

#### Tree build is now NlogN

N      |tbuild |tbmult
-------|-------|------
64     |0.010  |
512    |0.264  |26.40x
4096   |4.061  |15.38x
32768  |55.177 |13.59x
262144 |673.98 |12.22x
2097152|8601.5 |12.76x

#### Creating equivalent particles

N      |tree   |refine |equiv
-------|-------|-------|-----
512    |0.425  |0.531  |0.011
4096   |4.832  |2.578  |0.057
32768  |57.732 |20.550 |0.513
262144 |698.71 |158.22 |4.673

#### Time to find vels (mcycles) and rms error

N      |naive  |tree1  |err1   |tree2  |err2   |tree+refine
-------|-------|-------|-------|-------|-------|------------
64     |0.124  |0.123  |0.0    |0.123  |0.0    |0.014+0.085
512    |9.506  |4.857  |0.0    |4.864  |4.66e-5|0.397+0.994
4096   |663.583|332.68 |0.0    |112.4  |1.15e-3|4.681+2.442
32768  |53867.4|15384.1|7.89e-4|2108.08|9.16e-4|59.329+19.974
262144 |3193703|195733.|1.17e-3|24416.4|5.97e-4|711.77+161.47
2097152|1.99e+8|1.94e+6|8.99e-4|263423.|5.34e-4|8567.3+1310.7

#### Test accuracy vs. theta for equivalent particle treecode

All use N=262144 and itskip=34

theta  |tree2  |err2
-------|-------|-------
1.5    |76899.9|1.10e-4
1.2    |43780.8|2.65e-4
1.0    |29469.4|5.90e-4
0.9    |22837.6|1.08e-3
0.8    |14197.8|4.03e-3
0.7    |9628.09|1.09e-1

#### Performance tests on N=10^m

Treecode1 (1st order) uses theta=2.9, treecode2 (equivalent particles) uses theta=0.95.

N       | naive   |   tree1 |    err1 | tree2   | err2    | scaling | speedup
--------|---------|---------|---------|---------|---------|---------|--------
1000    | 39.73   | 18.47   | 0.0     | 11.97   | 2.02e-3 | na      | 3.32x
10000   | 1934.   | 1255.   | 1.12e-3 | 351.7   | 1.62e-3 | 29.4x   | 5.50x
100000  | 3.89e+5 | 2.75e+4 | 2.04e-3 | 6340.   | 2.19e-3 | 18.0x   | 61.4x
1000000 | 4.07e+7 | 4.16e+5 | 1.34e-3 | 8.65e+4 | 2.04e-3 | 13.6x   | 471x
10000000| 4.02e+9 | 5.75e+6 | 1.41e-2 | 1.23e+6 | 1.46e-2 | 14.2x   | 3270x

Performance asymptotes at 1.45 GFlop/s for direct summation, indicitave of non-SSE, single-threaded operation.

#### Scaling of O(N) solver

The O(N) solver logic is done, though the prolongation operation is inaccurate and the 
overall scheme seems slow. But it does seem to scale properly. Here are some performance figures.

N       | naive    | scale n  | tree2    | scale2   | error2   | fast     | scale    | error
--------|----------|----------|----------|----------|----------|----------|----------|---------
4096    | 6.500e+2 |          | 1.010e+2 |          | 1.497e-3 | 6.255e+2 |          | 1.104e-3
32768   | 6.925e+4 | 106.5x   | 1.554e+3 | 15.39x   | 1.200e-3 | 2.035e+4 | 32.53x   | 1.475e-2
262144  | 2.803e+6 | 40.48x   | 1.866e+4 | 12.01x   | 7.856e-4 | 1.704e+5 | 8.373x   | 1.780e-2
2097152 | 1.753e+8 | 62.54x   | 2.033e+5 | 10.90x   | 9.175e-4 | 1.586e+6 | 9.307x   | 1.791e-2

#### Behavior of O(N) solver

Setting box opening criterion to 1.0f and counting block-on-block operations, we have the following numbers. Leafs is the number of leaves in the tree (block size is 64). "ll per l" is the number of leaf-leaf interactions per leaf node. "bl per l" is the number of box-leaf interactions per leaf. And "b per b" is the number of source box or leaf interactions per non-leaf target box. One "interaction" here is 64 sources affecting 64 targets, whether equivalent or real.

N       | leafs | ll per l| bl per l| b per b
--------|-------|---------|---------|---------
4096    | 64    | 40.75   | 6.14062 | 4.80952
32768   | 512   | 67.5508 | 8.6543  | 18.4470
262144  | 4096  | 84.9561 | 14.5999 | 26.9609
2097152 | 32768 | 93.9102 | 16.6942 | 33.7703


## To do

* Start comparing accuracy of treecode and report it  - DONE
* Find out why tree-build is O(N^2) - DONE
* Why is error not zero for n=129 ? - DONE
* Write recursive equivalent-particle-finding routine - DONE
* Run with equivalent particles and compare performance and error - DONE
* Begin building target trees - DONE
* Work on O(N) method - DONE
* Increase accuracy of the prolongation operator
* Tweak box-opening criterion and see if it improves accuracy per time
* Use smarter or faster data structures in the O(N) list-building system
* Write O(N^2) method in double-precision to allow more fair error comparisons
  but only run it on the first 100 or 1000 target particles
* Get clang or g++ to vectorize the innermost loops
* Enable OpenMP parallelization - or - try std::thread
* Turn Particles and Tree into classes
* Specialize or Templatize the Particle class to allow it to be used efficiently for sources or targets
* Make the x,y,z particle coordinates into an array of axes, this might make it possible to use the same data structures for 2D or 4D tree codes, as well as more cleverly automating the tree split axis selection
* Start pulling the various algorithms (naive, tree1, tree2, fast) into separate...what, classes?

