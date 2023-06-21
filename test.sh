CUDA_VISIBLE_DEVICES='0'
exp_model_path="experiments/first-try/last.ckpt"
test_data_path="data/test.txt"
tag_path="data/tags.txt"
pretrained_model="/home/wl2020/pretrained-models/guwenbert-base"
predictions_path="predictions-first-try.txt"


python src/main.py  --eval \
                    --eval_batch_size 256 \
                    --test_data_path $test_data_path \
                    --tag_path $tag_path \
                    --pretrained_model $pretrained_model \
                    --predictions_path $predictions_path \
                    --exp_model_path $exp_model_path \
                    --lr 1e-5 \
                    --embedding_dim 768 \
                    --hidden_dim 256 \
                    --max_len 512 \
                    --random_seed 3407
                    
