all : onbody onvort2d

CPP=g++
CFLAGS=-std=c++11
OPTS=-O2 -march=native -fopenmp -ffast-math -ftree-vectorize -ftree-loop-vectorize

onbody : onbody.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<

onvort2d : onvort2d.cpp timing.h
	$(CPP) $(CFLAGS) $(OPTS) -o $@ $<

clean : 
	rm onbody onvort2d
