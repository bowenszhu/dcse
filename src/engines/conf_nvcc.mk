NVCC        		= nvcc
NVCCFLAGS   		= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS)" -O3 -arch=sm_75 --ptxas-options=-v
NVCCFLAGS_DEBUG   	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -G -arch=sm_75
NVCCFLAGS_PROFILE  	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -O3 -lineinfo) -arch=sm_75


