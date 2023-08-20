container_name=$1
key_scan=""

if [ -z "$1" ]
then
    container_name="node1"
fi

#if container_name is node1
# if [ "$container_name" = "node1" ]
# then
#     key_scan="ssh-keyscan -H node2 >> ~/.ssh/known_hosts"
# fi

docker run --rm --gpus all -it -v $PWD/mpi_test:/home/deepspeed/mpi_test --name $container_name --network deepspeed -p 2020:22 daynauth/mpi_test:latest