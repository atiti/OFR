CC=gcc
LDFLAGS=-lml -lcvaux -lhighgui -lcv -lcxcore
CFLAGS=-ggdb -I/usr/include/opencv


all: build

build:
	$(CC) $(CFLAGS) facerecog.c -o facerecog $(LDFLAGS)
	$(CC) $(CFLAGS) facedetect.c -o facedetect $(LDFLAGS)
