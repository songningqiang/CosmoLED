# For C code files

CC      = gcc-10
CCFLAGS = -Wall -c
CLFLAGS = -lm -o
CSLFLAGS = -shared -o

# The directory into which the stuff goes when compiled
# Note: EVERYTHING in this directory is killed by 'make clean' !!!

BINDIR = ../bin/

all:
	make lib.la
	make libod.so
	make libodNoComptonTau.so
	#make comphy.o
	#make od.o
	
clean:
	rm *.o *.la *.so
#	rm $(BINDIR)*
	
	
# ------ Libraries -----------------------------------------------------------#

lib.la: comphy.o od.o
	ar -cr $@ $^
	
libod.so: od.o
	$(CC) $(CSLFLAGS) $@ $^
	
libodNoComptonTau.so: odNoComptonTau.o
	$(CC) $(CSLFLAGS) $@ $^
    
# ------ Individual modules --------------------------------------------------#
comphy.o: comphy.c
	$(CC) $(CCFLAGS) $^
	
od.o: od.c
	$(CC) $(CCFLAGS) $^
	
odNoComptonTau.o: odNoComptonTau.c
	$(CC) $(CCFLAGS) $^
	
