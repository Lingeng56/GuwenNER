CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
checkpoint_path="experiments/first-try"
train_data_path="data/train.txt"
dev_data_path="data/validation.txt"
test_data_path="data/test.txt"
tag_path="data/tags.txt"
pretrained_model="/home/wl2020/pretrained-models/guwenbert-base"

mkdir -p $checkpoint_path

python src/main.py  --train \
                    --train_batch_size 4 \
                    --eval_batch_size 4 \
                    --train_data_path $train_data_path \
                    --dev_data_path $dev_data_path \
                    --test_data_path $test_data_path \
                    --tag_path $tag_path \
                    --checkpoint_path $checkpoint_path \
                    --pretrained_model $pretrained_model \
                    --min_epochs 1 \
                    --max_epochs 10 \
                    --lr 1e-5 \
                    --embedding_dim 768 \
                    --hidden_dim 256 \
                    --max_len 512 \
                    --freeze_bert \
                    --random_seed 3407
                    