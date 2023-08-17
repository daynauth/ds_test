container_name=$1
if [ -z "$1" ]
then
    container_name="node1"
fi

docker run --rm --gpus all -it -v $PWD/mpi_test:/home/deepspeed/mpi_test --name $container_name --network deepspeed -p 2020:20 daynauth/mpi_test:latest