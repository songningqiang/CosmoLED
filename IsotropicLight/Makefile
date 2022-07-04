# For C code files

CC      = gcc-10
CCFLAGS = -Wall -c
CLFLAGS = -lm -o
CSLFLAGS = -shared -o

# The directory into which the stuff goes when compiled
# Note: EVERYTHING in this directory is killed by 'make clean' !!!

BINDIR = ../bin/
SRCDIR = OpticalDepth/

all:
	make $(SRCDIR)lib.la
	make $(SRCDIR)libod.so
	make $(SRCDIR)libodNoComptonTau.so
	#make comphy.o
	#make od.o
	
clean:
	rm $(SRCDIR)*.o $(SRCDIR)*.la $(SRCDIR)*.so
#	rm $(BINDIR)*
	
	
# ------ Libraries -----------------------------------------------------------#

$(SRCDIR)lib.la: $(SRCDIR)comphy.o $(SRCDIR)od.o
	ar -cr $@ $^
	
$(SRCDIR)libod.so: $(SRCDIR)od.o
	$(CC) $(CSLFLAGS) $@ $^
	
$(SRCDIR)libodNoComptonTau.so: $(SRCDIR)odNoComptonTau.o
	$(CC) $(CSLFLAGS) $@ $^
    
# ------ Individual modules --------------------------------------------------#
comphy.o: $(SRCDIR)comphy.c
	$(CC) $(CCFLAGS) $^
	
od.o: $(SRCDIR)od.c
	$(CC) $(CCFLAGS) $^
	
odNoComptonTau.o: $(SRCDIR)odNoComptonTau.c
	$(CC) $(CCFLAGS) $^
	
