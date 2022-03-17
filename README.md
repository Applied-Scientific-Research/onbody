# onbody

Test C++ code for equivalent particle approximation technique and O(NlogN) and O(N) summation methods


## Summary

`onbody` is a testbed C++ library and driver program for a single evaluation of the forces
on a system of target points from a system of source particles (an "N-body" problem).

This is easiest understood in the context of gravitation: all stars (masses) in a galaxy 
attract all other stars; to find the forces between all of them on a computer, you would 
need to compute, for each star, the distance, direction, and then force that every other
star applies. This is a trivially easy algorithm to write, but it must perform about 
20 * N * N arithmetic operations to find all of the stars' new accelerations (where N is the
number of stars). This is called direct summation, and is an O(N^2) method.

In this software are "treecodes" that aim to make that calculation much faster, theoretically
C * N * log(N) operations, where C is some constant number (generally larger than 20).
Treecode algorithms are still relatively easy to program, but it is a little challenging to 
optimize their performance on current computer hardware.
Finally, there are methods (Fast Multipole Method, dual-tree-traversal, etc.) that 
can reduce that further to C * N operations (C is again a constant number, although 
it could be large), but at the cost of an even more intricate algorithm and more 
difficulty optimizing the code for peak performance.

The same mathematics and algorithms that work for gravitation problems also work for
a variety of other problems. Solving for electrostatic forces is identical to gravitation
with the exception that the "mass" of each source particle can be negative.
Similarly, incompressible fluid dynamics allows solutions where source particles
have a circulation property, and the same algorithms then solve for the resulting
velocity of those particles (called "vortex particle methods").

Each of the codes in this package will set up a system of source and target points and
perform the naive direct summation, three treecodes, and ultimately a "fast" (O(N)) 
summation of all particles for that particular problem.


## Build and run

    mkdir Release
    cd Release
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    ./ongrav3d -n=100000

Alternatively, configure the build to use another compiler with a command like:

    cmake -DCMAKE_CXX_COMPILER=/opt/gcc/gcc-6.3.0/bin/g++ -DCMAKE_BUILD_TYPE=Release ..

Note that the argument for the box-opening criterion (theta, also called the multipole acceptance
criterion "MAC") common to all treecodes is the reciprocal of what is commonly used.
Typically, theta=0.5 would mean that the source tree cell size is half of the distance
to the target point/cluster. In the codes here, that would be `-t=2` - we think of it as
the relative distance between the clusters is 2 times the cluster size.
A normal call to the gravitation solver would then look like this:

    ./ongrav3d -n=1000000 -t=2.0 -o=4

## Performance

The tests below were run on an AMD Ryzen 9 3950X 16 core processor, generally at ~4GHz.
The programs are multi-threaded with OpenMP, compiled with GCC 9.3.1 using `-Ofast -march=native`
and use a block size of 128.
All reported times are in wall-clock seconds reported with the high resolution timer from `std::chrono`.
Below is the performance of the `ongrav3d` program, coded to use charges instead of
masses (it is much harder to get high accuracy with + and - charges than with
always-positive masses), with my theta 1.11111 (MAC theta=0.9), 4th order interpolation
(5^3 Chebyshev points per tree node) and double-precision numbers for
storage and computation. RMS errors were around 1e-4.

N         | src tree | calc equivs |  direct  | pointwise | boxwise
----------|----------|-------------|----------|-----------|--------
10000     |  0.0020  |    0.0063   |  0.0137  |   0.0071  | 0.0092
100000    |  0.0099  |    0.0622   |  0.7850  |   0.1569  | 0.1365
1000000   |  0.1043  |    0.6065   |  413.46  |   3.0210  | 2.3598
10000000  |  1.7031  |    6.1892   |  43739.  |   47.929  | 36.797
100000000 |  20.830  |    62.212   |  5236300 |   639.84  | 461.25

![Performance vs. N, theta=0.9](doc/resNqd_t0p9.png)
![Performance vs. Error, varying order and theta](doc/res1Mqd_trad.png)

## Theory

#### VAM-split k-d trees
N-body methods with better than O(N^2) performance must take on the extra cost of
organizing the particles into spatial data structures called trees (hence "treecode").
Most still use spatial octrees (or quadtrees in two dimensions) where each node of
the tree is a cube (square) and is placed in space in a grid-like arrangement.
Unless particles are positioned uniformly (which is rare), each node in the tree has
a different number of particles. This can cause inefficiencies when performing 
the tree traversals and force calculations because GPUs and modern CPU hardware
is most efficient when performing arithmetic in larger, power-of-two blocks
(like 4, 8, and 16 on CPUs and 64, 128, and 256 on GPUs).

We began using VAM-split k-d trees in 2007 to achieve peak performance on GPUs, and
they have been the core of all of our treecodes since.
A VAMsplit k-d tree has the same number of particles in every leaf node *except for
the last one*, and every level of the tree has that same property.
Say we have 1000 particles in a VAM-split k-d tree with leaf node size of 256 (this is
very efficient for N-body methods on GPUs). The root of the tree has all 1000 points,
the next level is split such that the left side has 512, and the right has the remainder (488).
The next level is split so that all leaf nodes have 256 except the last, which has 232.
Indexing into such a tree is also easy, if the root node is index i, its children are
2\*i and 2\*i+1.
The main difference between this and a standard octree is that the tree nodes are not
arranged in a grid.

#### Partial sorting for fast tree building
Generating a spatial tree always involves sorting particles, generally along one 
of the coordinate axes (if a collection of particles is much larger in the x direction
than the y or z, we sort along x to find the first split).
But it's easy to waste time by performing a full sort when only a partial sort is
necessary. In the code here, we perform a very fast partial sort along the longest
coordinate axis such that the left (low) portion of the collection has blocksize \* 2^n
particles, where n depends on the level of the tree.
This uses an algorithm more like Floyd-Rivest than QuickSelect, as the problem is identical
to that of finding the k-th smallest element in an array (where k can be large).
The pivots are almost always close to those predicted by linear interpolation because
at each level, the node positions along that axis are effectively random.

#### Equivalent particles
Treecodes and fast codes must generate approximations of both the source terms on the 
underlying particles and the outputs on the target points, and this is most typically
done with multipole expansions. Thus, these codes have two code paths to compute the
influence of a block of particles on a target: multipoles for faraway source blocks, 
and direct summations for closer ones.
What if a code could use same easy-to-program direct-summation-style for both?
The origin of this software was an experiment in designing such a code.

The "equivalent particle" technique is where, given a `blockSize` representing the number of particles in all but one leaf node, every other node in the tree contains `blockSize` "equivalent particles" which are used to compute the influence of that box on target points. A simple treecode requires only the source points to have equivalent particles; and in this sense, they act like the multipole approximation of that tree node. For an O(N) method, the target tree also uses this system, so that all but one node in every level of the tree contains exactly `blockSize` equivalent target points, onto which the sources' influences are calculated, and from which those influences are passed down to its child boxes' equivalent target points. This means that every box-box, box-particle, or particle-box interaction can use the exact same computation kernel: `blockSize` (equivalent) particles affecting `blockSize` (equivalent) target points. This should simplify the programming of an O(N) fast summation code considerably, as long as appropriate trees and equivalent particles can be created efficiently.

#### Barycentric Lagrange form
Instead of generating N/2 equivalent particles from a tree node with N particles, we instead
create K^D proxy particles at Chebyshev nodes of the 2nd kind and interpolate the tree nodes'
particles onto those. Depending on K, this creates a higher-order distribution of charges/masses
with which to perform long-range interactions.
See [Wang-Tlupova-Krasny 2020](https://ieeexplore.ieee.org/abstract/document/9150146) for details.
In essence, this dramatically improves accuracy with only a small cost to performance.

To use the original equivalent-particle method, simply omit the "order" argument in the call:

    ./ongrav3d -n=1000000 -t=1

And to use the barycentric Lagrange interpolation, include it:

    ./ongrav3d -n=1000000 -t=1 -o=4

Note that the block size limits the available orders. Order 4 interpolation requires 125 nodes
(o+1)^D. The default block size is 128, so order 4 is the highest order available for D=3 solutions.
Two dimensional problems can go higher: order 10 requires 121 interpolation nodes.

    ./onvort2d -n=1000000 -t=1 -o=10

This restriction could be relaxed with some code changes.

## To Do

* Fix bugs hindering accuracy in `runNd` programs
* Finish implementing the barycentric Lagrange interpolator into the fast (O(N)) method
* Is it possible to use OpenGL to perform the tree walk and 64-on-64 evaluations? See [nvortexOpenGL](https://github.com/Applied-Scientific-Research/nvortexOpenGL) for sample code
* Use smarter or faster data structures in the O(N) list-building system
* Start pulling the various algorithms (naive, tree1, tree2, fast) into separate...what, classes?

## Credits

This program was written by Mark Stock with help and thanks to Prof. Krasny and collaborators for
the barycentric Lagrange interpolation method: [BaryTree on github](https://github.com/Treecodes/BaryTree/issues).

