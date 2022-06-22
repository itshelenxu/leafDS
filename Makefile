OPT?=3
VALGRIND?=0
SANITIZE?=0
CILK?=0


CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g  -std=c++20 -fmax-errors=1 -ftemplate-backtrace-limit=0 

LDFLAGS := 

ifeq ($(CILK),1)
CXX=clang++
CFLAGS += -fcilkplus
LDFLAGS += -L/home/ubuntu/xvdf_mounted/cilkrts/build/install/lib
endif
ifeq ($(SANITIZE),1)
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif

ifeq ($(OPT),3)
CFLAGS += -fno-signed-zeros  -freciprocal-math -ffp-contract=fast -fno-trapping-math  -ffinite-math-only
ifeq ($(VALGRIND),0)
CFLAGS += -march=native #-static
endif
endif


VERIFY_COUNT ?= 10000


INCLUDES := leafDS.hpp 
.PHONY: all clean tidy

all:  basic test 
#build_profile profile opt


basic: test.cpp leafDS.hpp StructOfArrays/SizedInt.hpp StructOfArrays/soa.hpp
	$(CXX) $(CFLAGS) $(DEFINES) test.cpp $(LDFLAGS) -o basic


test: basic
	@mkdir -p test_out
	@./basic --update_test --el_count $(VERIFY_COUNT) --verify  >test_out/update_test || (echo "./basic --update_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@./basic --update_values_test --el_count $(VERIFY_COUNT) --verify  >test_out/update_values_test || (echo "./basic --update_values_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@wait
	@sleep 1
	@echo "Tests Finished"


clean:
	rm -f basic
