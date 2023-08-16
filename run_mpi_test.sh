benchmark=$PWD/benchmark
mpi_test=$PWD/mpi_test

docker run --rm --gpus all -it -v $mpi_test:/home/deepspeed/mpi_test --name node1 --network deepspeed mpi_test