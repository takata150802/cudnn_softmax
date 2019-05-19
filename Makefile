CUDNN_PATH = /usr/include

main: main.cu
	nvcc -g \
        -O0 \
        -I$(CUDNN_PATH) \
        -lcudnn \
        -std=c++11 \
        main.cu \
        -o main.out 
