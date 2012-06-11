CC=gcc
LDFLAGS=-lml -lcvaux -lhighgui -lcv -lcxcore
CFLAGS=-ggdb -I/usr/include/opencv


all: build

build:
	$(CC) $(CFLAGS) facedetect.c -o facedetect $(LDFLAGS)
