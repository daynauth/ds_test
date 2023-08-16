benchmark=$PWD/benchmark

docker run --rm --gpus all -it -v ${benchmark}:/home/deepspeed/benchmark deepspeed