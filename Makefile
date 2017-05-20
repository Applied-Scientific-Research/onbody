all : onbody

CPP=g++
CFLAGS=-std=c++11
OPTS=-O2 -march=native -fopenmp -ffast-math -ftree-vectorize -ftree-loop-vectorize

onbody : onbody.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<

clean : 
	rm onbody
