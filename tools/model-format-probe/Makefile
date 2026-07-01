CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

.PHONY: all clean

all: model-format-probe

model-format-probe: model-format-probe.c
	$(CC) $(CFLAGS) -o $@ model-format-probe.c

clean:
	rm -f model-format-probe
