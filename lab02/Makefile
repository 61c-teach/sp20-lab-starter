UNAME_S := $(shell uname -s)
CC=/*YOUR CODE HERE*/
LD=/*YOUR CODE HERE*/
CFLAGS=/*YOUR CODE HERE*/
LDFLAGS=

ifeq ($(UNAME_S), Darwin)
    MEMCHECK=valgrind --tool=memcheck --leak-check=full --track-origins=yes --dsymutil=yes --suppressions=osx_vector.supp
endif

ifeq ($(UNAME_S), Linux)
    MEMCHECK=valgrind --tool=memcheck --leak-check=full --track-origins=yes
endif

LFSR_OBJS = lfsr.o test_lfsr.o
LFSR_PROG = lfsr

BINARIES=vector-test bit_ops lfsr_prog

all: $(BINARIES)

bit_ops: bit_ops.o test_bit_ops.o
	$(CC) $(CFLAGS) -g -o bit_ops bit_ops.o test_bit_ops.o

lfsr: $(LFSR_OBJS)
	$(CC) $(CFLAGS) -g -o /*YOUR CODE HERE*/

vector-test: /*YOUR CODE HERE*/
	/*YOUR CODE HERE*/

lfsr.c: lfsr.h
test_lfsr.c: lfsr.h

bit_ops.c: bit_ops.h
test_bit_ops.c: bit_ops.h

.c.o:
	$(CC) -c $(CFLAGS) $<

vector-memcheck: vector-test
	$(MEMCHECK) ./vector-test

clean: 
	-rm -rf core *.o *~ "#"*"#" Makefile.bak $(BINARIES) *.dSYM
        
vector.c: vector.h
vector-test.c: vector.h
