# onbody
Test C++ code for equivalent particle approximation technique and O(NlogN) and O(N) summation methods

Block size 64
Single threaded
MacOS 10.11

#### 1st order treecode (boxes are singular particles)

N       naive   nmult   tree=3  tmult   tbuild  tbmult
64      0.127           0.127           0.013
512     9.371   73.79x  4.894   38.18x  0.424   32.62x
4096    625.77  66.78x  167.29  34.50x  5.103   12.03x
32768   48637.  77.72x  2573.1  15.38x  113.76  22.29x
262144  35625+2 73.25x  30103.  11.70x  7391.2  64.97x
2097152 na      na      313147  10.40x  643415  87.05x
big run took 138MB
tree is O(N^2) or worse - why? was std:copy on full arrays

