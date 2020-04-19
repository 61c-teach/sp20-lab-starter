CC=gcc
CFLAGS=-fopenmp -g -std=gnu99 -ggdb3 -c -Wall
LDFLAGS=-fopenmp
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLES=v_add dotp hello server_basic server_process

.PHONY: all clean

all: $(EXECUTABLES)

v_add: v_add.o
hello: hello.o
dotp: dotp.o
server_basic: server_basic.o libhttp.o omp_apps.o server_utils_seriel.o libbmp.o
server_process: server_process.o libhttp.o omp_apps.o server_utils_process.o libbmp.o

$(EXECUTABLES):
	$(CC) $(LDFLAGS) $^ -o $@

server_basic.o: server.c libhttp/libhttp.c omp_apps.c server_utils.c libbmp/libbmp.c
server_process.o: server.c libhttp/libhttp.c omp_apps.c server_utils.c libbmp/libbmp.c
omp_app.o: omp_apps.c libbmp/libbmp.c


server_utils_seriel.o: server_utils.c
	$(CC) $(CFLAGS) -c $< -o $@
server_utils_process.o: server_utils.c
	$(CC) -DPROC $(CFLAGS) -c $< -o $@
server_process.o:
	$(CC) $(CFLAGS) -c $< -o $@
server_basic.o:
	$(CC) $(CFLAGS) -c $< -o $@

libbmp.o:
	$(CC) $(CFLAGS) -c libbmp/libbmp.c -o $@
libhttp.o:
	$(CC) $(CFLAGS) -c libhttp/libhttp.c -o $@

omp_app.o: libbmp.o
	$(CC) $(CFLAGS) -c omp_apps.c -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLES) *.o
