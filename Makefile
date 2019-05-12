CUDNN_PATH = /usr/include

main: main.cu
	nvcc -g -O0 -I$(CUDNN_PATH) -lcudnn main.cu -o main.out 
