all : ongrav3d onvort2d

CPP=g++
CFLAGS=-std=c++11
OPTS=-O2 -march=native -fopenmp -ffast-math -ftree-vectorize -ftree-loop-vectorize

ongrav3d : ongrav3d.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<

onvort2d : onvort2d.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<

clean : 
	rm ongrav3d onvort2d
