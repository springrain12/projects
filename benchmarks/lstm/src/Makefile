CC=g++
CFLAGS=-g -Wall
LDFLAGS=
RM=rm -rf

SRCS=$(wildcard *.cc)
HDRS=$(wildcard *.h)
OBJS=$(SRCS:.cc=.o)
EXEC=lstm

default: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.cc $(HDRS)
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	$(RM) $(OBJS) $(EXEC)

