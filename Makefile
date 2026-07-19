# stabkernel — multi-arch GF(2) / stabilizer bit-linear-algebra kernels
# Detects host arch and builds the matching hand-written assembly tier.

CC      ?= cc
# -fopenmp powers the multi-core layer. Build with `make NATIVE=1` to also add
# -march=native -funroll-loops and squeeze the most out of the host CPU.
CFLAGS  ?= -O3 -Wall -Wextra -Iinclude -fopenmp
ifdef NATIVE
CFLAGS  += -march=native -funroll-loops
endif
ARCH    := $(shell uname -m)

COMMON_SRC := src/gf2_scalar.c src/dispatch.c src/gf2_linalg.c src/wrappers.c src/gf2_parallel.c

ifneq (,$(filter x86_64 amd64,$(ARCH)))
  ASM_SRC := src/gf2_x86.S
  ARCHNAME := x86-64 (AVX-512 / AVX2 / scalar)
endif
ifneq (,$(filter aarch64 arm64,$(ARCH)))
  ASM_SRC := src/gf2_arm.S
  ARCHNAME := aarch64 (NEON / scalar)
endif
ASM_SRC ?=
ARCHNAME ?= $(ARCH) (scalar only)

SRC := $(COMMON_SRC) $(ASM_SRC)

.PHONY: all lib test bench clean info
all: info lib test bench

info:
	@echo "Building stabkernel for: $(ARCHNAME)"

lib: libstabkernel.so
libstabkernel.so: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRC)

test: test_correctness
	./test_correctness
test_correctness: $(SRC) tests/test_correctness.c
	$(CC) $(CFLAGS) -o $@ $(SRC) tests/test_correctness.c

bench: bench_run bench_parallel
bench_run: $(SRC) bench/bench.c
	$(CC) $(CFLAGS) -o $@ $(SRC) bench/bench.c
bench_parallel: $(SRC) bench/bench_parallel.c
	$(CC) $(CFLAGS) -o $@ $(SRC) bench/bench_parallel.c

# Multi-core scaling report (1 -> all cores).
scale: bench_parallel
	./bench_parallel

clean:
	rm -f libstabkernel.so test_correctness bench_run bench_parallel
