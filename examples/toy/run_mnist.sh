mkdir data
cd data
wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz 
cd ..
CUDA_VISIBLE_DEVICES=0,1 deepspeed pipeline_toy.py