# onbody
Test C++ code for equivalent particle approximation technique and O(NlogN) and O(N) summation methods

## Summary

`onbody` is a testbed C++ program for a single evaluation of the gravitational forces
of a set of particles on a system of target points. Key goals are to learn:

* How C++ data structures and algorithms will work for this problem
* Whether a system that uses equivalent particles at all levels of a tree can generate solutions of sufficient accuracy
* Whether using a hierarchical system of equivalent target points allows creation of an O(N) fast summation method of sufficient accuracy and low programmer effort

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

N       tree    refine  equiv
512     0.425   0.531   0.011
4096    4.832   2.578   0.057
32768   57.732  20.550  0.513
262144  698.71  158.22  4.673

#### Time to find vels (mcycles) and rms error

N       naive   tree1   err1    tree2   err2    tree+refine
64      0.124   0.123   0.0     0.123   0.0     0.014+0.085
512     9.506   4.857   0.0     4.864   4.66e-5 0.397+0.994
4096    663.583 332.68  0.0     112.4   1.15e-3 4.681+2.442
32768   53867.4 15384.1 7.89e-4 2108.08 9.16e-4 59.329+19.974
262144  3193703 195733. 1.17e-3 24416.4 5.97e-4 711.77+161.47
2097152 1.99e+8 1.94e+6 8.99e-4 263423. 5.34e-4 8567.3+1310.7

#### Test accuracy vs. theta for equivalent particle treecode

All use N=262144 and itskip=34

theta   tree2   err2
1.5     76899.9 1.10e-4
1.2     43780.8 2.65e-4
1.0     29469.4 5.90e-4
0.9     22837.6 1.08e-3
0.8     14197.8 4.03e-3
0.7     9628.09 1.09e-1

## To do

* Start comparing accuracy of treecode and report it  - DONE
* Find out why tree-build is O(N^2) - DONE
* Why is error not zero for n=129 ? - DONE
* Write recursive equivalent-particle-finding routine - DONE
* Run with equivalent particles and compare performance and error - DONE
* Begin building target trees - DONE
* Work on O(N) method
* Write O(N^2) method in double-precision to allow more fair error comparisons
  but only run it on the first 100 or 1000 target particles

