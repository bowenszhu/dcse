NVCC        		= nvcc

# Gencode arguments
SMS ?= 35 37 50 52 60 61 70 75 80 86

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

NVCCFLAGS   		= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS)" -O3 --ptxas-options=-v
NVCCFLAGS_DEBUG   	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -G
NVCCFLAGS_PROFILE  	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -O3 -lineinfo)


