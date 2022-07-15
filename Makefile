CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=
LDFLAGS=
LDLIBS=-lOpenCL

SRCS=matrix_multiply.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: matrix_multiply

matrix_multiply: $(OBJS)
	$(CXX) $(LDFLAGS) -o matrix_multiply $(OBJS) $(LDLIBS)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) tool
