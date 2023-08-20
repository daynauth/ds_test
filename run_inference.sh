container_name=$1

benchmark=$PWD/benchmark

if [ -z "$1" ]
then
    container_name="node1"
fi

docker run --rm --gpus all -it -v ${benchmark}:/home/deepspeed/benchmark --network deepspeed --name $container_name daynauth/deepspeed