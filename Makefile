KOKKOS_PATH ?= ${HOME}/Kokkos/kokkos
KOKKOSKERNELS_PATH ?= ${HOME}/Kokkos/kokkos-kernels

KOKKOS_DEVICES=OpenMP
KOKKOS_ARCH = "SNB,Kepler35"

# Turn of ETI
KOKKOSKERNELS_SCALARS =  

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)/src/*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)/src/*.hpp)

KOKKOS_CUDA_OPTIONS=enable_lambda,force_uvm

default: build
	echo "Start Build"

CXX = mpicxx

LINK = ${CXX}

CXXFLAGS = -O3 -g 
override CXXFLAGS += -I$(MAKEFILE_PATH)/src/
LINKFLAGS = -O3 -g 

EXE = hpcg.kokkos
DEPFLAGS = -M

vpath %.cpp $(sort $(dir $(SRC)))

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos
include ${KOKKOSKERNELS_PATH}/Makefile.kokkos-kernels

$(warning $(OBJ) $(EXE) $(sort $(dir $(SRC))))

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(KOKKOSKERNELS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(KOKKOSKERNELS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOSKERNELS_CPPFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

