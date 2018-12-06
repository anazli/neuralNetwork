# Makefile for simulation's 2 executable (main)
EXEDIR=.
all : $(EXEDIR)/main $(EXEDIR)/multi_classification 
# The compiler
CC=g++

#Variables
DEBUG=-ggdb
OPTIM=-O3
ISO=gnu++14
PAR=-fopenmp

IDIR=include
ODIR=obj
#SDIR=src
VPATH=src include

# Compiler flags:
CFLAGS=-I$(IDIR) 
FLAGS= $(DEBUG) -std=$(ISO)
LFLAGS=-I$(ODIR)

#Linking object files
_DEPS = matrix.h model.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ1 = main.o matrix.o model.o
OBJ1 = $(patsubst %,$(ODIR)/%,$(_OBJ1))
_OBJ2 = multi_classification.o matrix.o model.o
OBJ2 = $(patsubst %,$(ODIR)/%,$(_OBJ2))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) $(FLAGS) -c -o $@ $< $(CFLAGS) 

$(EXEDIR)/main: $(OBJ1)
	$(CC) $(FLAGS) -o $@ $^ $(LFLAGS) 
$(EXEDIR)/multi_classification: $(OBJ2)
	$(CC) $(FLAGS) -o $@ $^ $(LFLAGS) 

.PHONY: clean

clean :
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
