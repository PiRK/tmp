hello: hello.o mymax.o
	gcc -o hello hello.o mymax.o

mymax.o: mymax.c
	gcc -o mymax.o -c mymax.c -W -Wall -ansi -pedantic

hello.o: hello.c mymax.h
	gcc -o hello.o -c hello.c -W -Wall -ansi -pedantic