CXX         	= g++
CXXFLAGS    	= -O3 -fPIC -march=native -std=c++11 -Wno-deprecated-declarations -Wno-attributes
CXXFLAGS_DEBUG	=  -O0 -g -rdynamic -fPIC -march=native -std=c++11 -Wno-deprecated-declarations -Wno-attributes

CC          	= gcc
CFLAGS      	= -O3 -march=native -fPIC 

ARCH        	= /usr/bin/ar
ARCHFLAGS   	= cr
RANLIB      	= /usr/bin/ranlib

OMP_FLAG    	= -fopenmp
