include conf_gcc.mk
include conf_nvcc.mk

TARGET      = engines.so
IFLAGS      = -I./src -I./pybind11/include/pybind11 `python3-config --includes`
IFLAGS_GPU  = $(IFLAGS) -I/usr/local/cuda/include
DFLAGS      = 
DFLAGS_GPU	= $(DFLAGS) -DWITH_GPU
LFLAGS_BASE	+= 
LFLAGS      = $(LFLAGS_BASE)
LFLAGS_MT  	= $(LFLAGS_BASE) 
LFLAGS_GPU 	= $(LFLAGS_BASE) -lcusparse -lcublas

BUILD_DIR	= build

SUBDIRS     = 
BUILD_DIRS  = $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(SUBDIRS))


OBJ =  operator_set_interpolator_static.o \
	py_evaluator_iface.o \
	py_globals.o \
	py_main.o \
	py_operator_set_interpolator_rates.o \
	py_operator_set_interpolator_static.o \
	py_piecewise_linear_interpolator_static.o \
	py_piecewise_linear_interpolator_adaptive.o \
	engines_build_info.o

OBJ_GPU =  	operator_set_interpolator_gpu.cu.o \
			py_operator_set_interpolator_gpu.o \
			PiecewiseLinearInterpolator_static_cuda.cu.o \
            py_piecewise_linear_interpolator_static_cuda.cu.o


OBJ_D = $(patsubst %,$(BUILD_DIR)/%,$(OBJ))
OBJ_GPU_D = $(patsubst %.o,$(BUILD_DIR)/%.gpu.o,$(OBJ_GPU) $(OBJ))
OBJ_MT_D = $(patsubst %.o,$(BUILD_DIR)/%.mt.o,$(OBJ))

OBJ_DEBUG_D = $(patsubst %.o,$(BUILD_DIR)/%.od,$(OBJ))
OBJ_GPU_DEBUG_D = $(patsubst %.o,$(BUILD_DIR)/%.gpu.d.o,$(OBJ_GPU) $(OBJ))
OBJ_MT_DEBUG_D = $(patsubst %.o,$(BUILD_DIR)/%.mt.od,$(OBJ))

BUILD_MACHINE := $(shell whoami)@$(shell hostname)
BUILD_DATE := $(shell TZ=Europe/Amsterdam date +"%d/%m/%Y, %H:%M:%S")
BUILD_GIT_HASH := $(shell git describe --always --dirty --match 'NOT A TAG')

dir_guard=@mkdir -p $(@D)


default: $(TARGET)


.PHONY: gpu mt clean clean_target build_dirs build_info

# Additional rule to force rebuild build_info.cpp file every time
src/engines_build_info.cpp: build_info
	$(shell echo "#include \"engines_build_info.h\""  > src/engines_build_info.cpp)
	$(shell echo "const char *ENGINES_BUILD_DATE = \"$(BUILD_DATE)\";"  >> src/engines_build_info.cpp)
	$(shell echo "const char *ENGINES_BUILD_MACHINE = \"$(BUILD_MACHINE)\";"  >> src/engines_build_info.cpp)
	$(shell echo "const char *ENGINES_BUILD_GIT_HASH = \"$(BUILD_GIT_HASH)\";"  >> src/engines_build_info.cpp)
	cat src/engines_build_info.cpp

.PHONY: gpu mt debug gpu_debug mt_debug clean clean_target build_dirs 

build_dirs: $(BUILD_DIRS)

$(BUILD_DIRS):
	mkdir -p $@


$(BUILD_DIR)/%.o:          src/%.cpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(DFLAGS)  -c -o  $@ $<

$(BUILD_DIR)/%.gpu.o:          src/%.cpp
	$(CXX) $(CXXFLAGS) $(IFLAGS_GPU) $(DFLAGS_GPU)  -c -o  $@ $<

$(BUILD_DIR)/%.cu.gpu.o:          src/%.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(IFLAGS_GPU) $(DFLAGS_GPU) -c -o  $@ $<
	
$(BUILD_DIR)/%.mt.o:          src/%.cpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $(IFLAGS) $(DFLAGS)  -c -o  $@ $<

$(BUILD_DIR)/%.od:          src/%.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $(IFLAGS) $(DFLAGS)  -c -o  $@ $<

$(BUILD_DIR)/%.gpu.d.o:          src/%.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $(IFLAGS_GPU) $(DFLAGS_GPU)  -c -o  $@ $<

$(BUILD_DIR)/%.cu.gpu.d.o:          src/%.cu
	$(NVCC) $(NVCCFLAGS_DEBUG) $(GENCODE_FLAGS) $(IFLAGS_GPU) $(DFLAGS_GPU) -c -o  $@ $<
	
$(BUILD_DIR)/%.mt.od:          src/%.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $(OMP_FLAG) $(IFLAGS) $(DFLAGS)  -c -o  $@ $<



$(TARGET): build_dirs $(OBJ_D) $(LIN_LIB)
	$(CXX) -shared -o $(TARGET) $(OBJ_D) $(LFLAGS)

gpu: build_dirs $(OBJ_GPU_D) $(LIN_LIB)
	$(NVCC) -shared -o $(TARGET) $(OBJ_GPU_D) $(LFLAGS_GPU)

mt: build_dirs $(OBJ_MT_D) $(LIN_LIB)
	$(CXX) -shared -o $(TARGET) $(OBJ_MT_D) $(LFLAGS_MT) $(OMP_FLAG)

debug: build_dirs $(OBJ_DEBUG_D) $(LIN_LIB)
	$(CXX) -shared -o $(TARGET) $(OBJ_DEBUG_D) $(LFLAGS)

gpu_debug: build_dirs $(OBJ_GPU_DEBUG_D) $(LIN_LIB)
	$(NVCC) -shared -o $(TARGET) $(OBJ_GPU_DEBUG_D) $(LFLAGS_GPU)

mt_debug: build_dirs $(OBJ_MT_DEBUG_D) $(LIN_LIB)
	$(CXX) -shared -o $(TARGET) $(OBJ_MT_DEBUG_D) $(LFLAGS_MT)



clean:
	rm -rf $(TARGET) $(OBJ_D) $(OBJ_MT_D) $(OBJ_GPU_D) $(OBJ_DEBUG_D) $(OBJ_MT_DEBUG_D) $(OBJ_GPU_DEBUG_D)

clean_target:
	rm -rf $(TARGET)
