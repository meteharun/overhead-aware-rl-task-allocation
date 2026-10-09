# Build the measurement harness on the board.
#
#   make                               native build, on the i.MX95 itself
#   make CC=aarch64-linux-gnu-gcc      cross build from a host
#
# -O2 and not -O3: the production build is -O2 and the number should describe
# the code that runs, not the fastest code the compiler can be talked into.
#
# The task bodies are statcom_tasks.c, the 25 tasks the campaign measured.

CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
LDFLAGS ?= -lm -lrt

BIN  = measure_wcet
OBJS = measure_wcet.o statcom_tasks.o

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

measure_wcet.o: measure_wcet.c
	$(CC) $(CFLAGS) -c $< -o $@

statcom_tasks.o: statcom_tasks.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(BIN) $(OBJS)

.PHONY: all clean
