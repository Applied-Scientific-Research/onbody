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

    mkdir Release
    cd Release
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    ./onbody -n=10000

Alternatively, configure the build to use another compiler with a command like:

    cmake -DCMAKE_CXX_COMPILER=/opt/gcc/gcc-6.3.0/bin/g++ -DCMAKE_BUILD_TYPE=Release ..

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

#### GCC 4.9.2 vs. 5.4.0 vs. 6.3.0

It seems very hard to get gcc 4.9.2 to vectorize loops. Here are some performance numbers from a 6-core AMD Phenom II X6 1090T processor at 3.21 GHz (OpenMP active) using `g++` 4.9.2, 5.4.0, and then 6.3.0. The command-line to build both versions was:

    g++ -std=c++11 -O2 -march=native -fopenmp -ffast-math -ftree-vectorize -ftree-loop-vectorize -o onbody onbody.cpp

N       | naive  | tree2  | fast
--------|--------|--------|--------
10000   | 1202.3 | 240.78 | 733.52
100000  | 111748 | 3403.1 | 14880
1000000 | 1.157+7| 46156  | 198450

N       | naive  | tree2  | fast
--------|--------|--------|--------
10000   | 378.81 | 109.20 | 286.29
100000  | 38273. | 1422.3 | 5499.6
1000000 | 4448477| 20124  | 72884

N       | naive  | tree2  | fast
--------|--------|--------|--------
10000   | 224.50 | 75.518 | 199.17
100000  | 23761. | 987.29 | 3489.9
1000000 | 4270680| 13004  | 45853

It looks like both 5.4.0 and 6.3.0 can vectorize properly, though 6.3.0 is substantially better. Note that CUDA 8.0 can only work with gcc versions 5.x or older.

Also note that the `tree2` method, a simple treecode, can solve for the velocities from 1 million vortex particles to 1.5e-3 mean error in four seconds (13004/3210) on a CPU alone. The OmegaFlow v2 code does the same in about 20s on the CPU and 0.7s when using a 1060GTX GPU.

#### How much work is the treecode doing?

For ~2e-3 error (theta=0.95), the equivalent particle treecode performs, on average, and for each target 
particle, the following number of box interactions. Also are the same numbers for the O(N) code for
numbers of source boxes per target box with theta=1.6.

N       | leaf   | equiv   | leaf-b  | equiv-b
--------|--------|---------|---------|--------
10000   | 7.8352 | 20.5085 | 74.2102 | 3.55414
100000  | 10.066 | 37.0454 | 142.049 | 5.0032
1000000 | 11.043 | 53.5838 | 194.929 | 6.53357

Clearly the logic or the box-opening criteria for the O(N) solver are flawed, or the prolongation
is horribly inaccurate.

#### Performance on 12-core pair of E5-2640

Runs at 2.8 GHz when all-cores are going, 3 GHz when one. First, GCC 4.8.5 (which has no `-ftree-loop-vectorize` option):

N        | build src tree | naive     | tree2   | fast
---------|----------------|-----------|---------|---------
10000    | 30.200         | 418.651   | 77.973  | 241.923
100000   | 300.189        | 33630.274 | 1177.29 | 4255.02
1000000  | 3165.784       | 3404879.2 | 14892.9 | 56113.8
10000000 | 47878.648      | 341073932 | 223259.7| 683045.8

Now 4.9.2, which has both vectorize options:

N        | build src tree | naive     | tree2   | fast
---------|----------------|-----------|---------|---------
100000   | 300.037        | 33089.904 | 1285.03 | 4150.51
1000000  | 3468.261       | 3347075.6 | 14747.3 | 54929.4

Now 5.4.0:

N        | build src tree | naive     | tree2   | fast
---------|----------------|-----------|---------|---------
100000   | 294.262        | 16795.127 | 667.680 | 2165.22
1000000  | 3166.409       | 1687193.3 | 7775.89 | 28453.8

And 6.3.0:

N        | build src tree | naive     | tree2   | fast
---------|----------------|-----------|---------|---------
100000   | 285.601        | 5575.831  | 318.697 | 903.100
1000000  | 3124.428       | 626915.86 | 2961.90 | 10244.95

And finally 7.1.0 (yep, still 2.8 GHz):

N        | build src tree | naive     | tree2   | fast
---------|----------------|-----------|---------|---------
100000   | 298.209        | 5468.109  | 310.892 | 852.453
1000000  | 3183.098       | 659180.43 | 2884.31 | 9562.412
10000000 | 48107.493      | 81118117. | 43595.8 | 118644.39

For the n=100000 problem, this CPU achieves 107 GFlop/s.

#### Parallelizing the tree-build

Using GCC 6.3.0 we add OpenMP tasks to parallelize the tree-build. Here are some performance results for the sources in `ongrav3d`. All measurements are in mcycles on `spy` using `blockSize` of 64.

N        | build tree serial | build tree omp | refine serial | refine omp
---------|-------------------|----------------|---------------|------------
100000   | 186.627           | 70.526         | 41.942        | 10.050
1000000  | 2740.769          | 920.562        | 396.547       | 87.359
10000000 | 48629.241         | 22333.097      | 3998.997      | 887.707

Using `std::async` to partition the first few levels of tree-builds saves a little more time. These are running on a pair of 6-core, 2.9 GHz E5-2640 CPUs (2.5 GHz nominally), and built with GCC 6.3.0.

N        | build tree | build tree async
---------|------------|------------------
100000   | 111.375    | 83.539
1000000  | 1000.821   | 690.409
10000000 | 20730.938  | 14150.123

Now that we are using std::chrono for timing, let's look at these performance numbers in a sane unit - seconds. These were conducted on an 8-core AMD Ryzen 2700X CPU. Thetas are 2.75 for both fast codes, and RMS errors peaked at 5e-4.

N        | src tree | calc equivs |  O(N^2) | O(NlogN) | O(N)
---------|----------|-------------|---------|----------|-------
10000    |  0.0080  |    0.0024   |  0.0288 |   0.0192 | 0.0392
100000   |  0.0198  |    0.0068   |  2.5251 |   0.4484 | 1.1426
1000000  |  0.1708  |    0.0384   |  268.51 |   8.7608 | 18.696
10000000 |  3.2547  |    0.3782   |  38036. |   227.87 | 213.30

This is the same table but for an 8-core Intel i7-5960X Haswell CPU and GCC 7.3.0.

N        | src tree | calc equivs |  O(N^2) | O(NlogN) | O(N)
---------|----------|-------------|---------|----------|-------
10000    |  0.0113  |    0.0011   |  0.0232 |   0.0200 | 0.0419
100000   |  0.0303  |    0.0074   |  2.0405 |   0.4446 | 0.9111
1000000  |  0.2042  |    0.0475   |  236.83 |   8.1706 | 14.556
10000000 |  3.5744  |    0.4113   |  21307. |   127.70 | 164.02

#### Current treecode performance
Using the `bhmain` driver for Barnes-Hut interactions in 2D on a i7-7500U, with GCC 9.2.1 compiling the code with `-O2 -march=native -ffast-math -ftree-vectorize -ftree-loop-vectorize` flags, and using theta of 1.3, we have the following times reported from chrono (best out of 3).

N        |block=16|   32   |   64   |   128    
---------|--------|--------|--------|--------
10000    | 0.0431 | 0.0427 | 0.0461 | 0.0477
100000   | 0.1393 | 0.1748 | 0.1921 | 0.2316
1000000  | 2.3903 | 2.6985 | 3.3061 | 4.1958

New compilation options are `-O3 -march=native -ffast-math`, also theta=4.0, blockSize=32, and forcing dynamic scheduling for the treecodes. The 2p machine is the above i7-7500U, the 8p is an i7-5960X running Fedora 29, compiled with GCC 8.3.0.

N        | omp 2p | omp 8p | memory
---------|--------|--------|--------
1000     | 0.0044 | 0.0068 |
10000    | 0.0502 | 0.0099 |
100000   | 0.4514 | 0.2198 |
1000000  | 6.6784 | 2.684  |
10000000 | 95.302 | 38.74  | 2.6 GB
20000000 |        | 88.96  | 5.1 GB
50000000 |        | 232.1  | 11.6 GB

## To Do

* Is it possible to use OpenGL to perform the tree walk and 64-on-64 evaluations? See [nvortexOpenGL](https://github.com/Applied-Scientific-Research/nvortexOpenGL) for sample code
* Is it possible to pull [Vc](https://github.com/VcDevel/Vc) into this project?
* Increase accuracy of the prolongation operator - this means writing a simple linear least squares solver to determine the solution and gradient at a child point, given a set of weighted parent neighbor (equivalent point) values
* Use a nearest-neighbor search for the prolongation - don't just take the other 7 or 15 in the parent box; but that uses non-local information!
* Pull in Eigen to assemble and solve the matrix equation for the prolongation - we need more moments taken into consideration to raise the accuracy, I believe
* Use smarter or faster data structures in the O(N) list-building system
* Start pulling the various algorithms (naive, tree1, tree2, fast) into separate...what, classes?
* Add radii to the target points (even if all zeros) and include their effect in the core function
* Support different core functions; right now we only use the Rosenhead-Moore kernel

