all : onbody

CPP=g++
CFLAGS=-std=c++11
OPTS=-O2 -march=native -fopenmp -ffast-math

onbody : onbody.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<
