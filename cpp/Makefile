CC=gcc
NVCC=nvcc
CFLAGS=-O3 -Wall -Wextra -ansi -std=c99 -pedantic
OBJ=obj/*.o
IDIR=../include/

all:
	mkdir -p lib
	mkdir -p obj
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/*.c -I${IDIR}
	${CC} -shared -o lib/libgraphfl.so ${OBJ} -lgsl -lgslcblas

conda3:
	mkdir -p lib
	mkdir -p obj
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/*.c -I/opt/anaconda3/envs/shadow/include/ -I${IDIR}
	${CC} -shared -o lib/libgraphfl.so ${OBJ} -L/opt/anaconda3/envs/shadow/lib/ -lgsl -lgslcblas
	
docs:
	doxygen Doxyfile

clean:
	rm obj/*.o

cleanall:
	rm obj/*.o
	rm lib/*.so
