echo "Enter with the number of nodes you want to use: "
read nodes

echo "Enter with the number of GPUs you want to use: "
read ngpus

echo "Enter with the number of epochs you want to train: "
read nepochs

echo "Enter with the start epoch (checkpoint): "
read startepoch

python3 src/make_dataset.py --datadir \
                            --gtruthdir \
                            --destdir 

torchrun --nnodes=$nodes \
    --nproc_per_node=$ngpus \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0\
    src/train.py --datadir $HOME/Downloads/raw/REAL_DROPLETS \
                 --gtruthdir $HOME/Downloads/raw/REAL_DROPLETS \
                 --cpdir \
                 --ngpus $ngpus \
                 --nepochs $nepochs \
                 --startepoch $startepoch

