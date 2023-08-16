#MODEL='gpt2'



MODELS=('gpt2' 'gpt2-xl' 'bert-base' 'bert-large')

echo "model,deepspeed,huggingface" >> results.csv
for MODEL in "${MODELS[@]}"; do
    CMD="deepspeed infer_deepspeed.py --model $MODEL --ds"
    ds_time=$(eval $CMD | grep 'Average time' | awk '{print $6}')

    CMD="deepspeed infer_deepspeed.py --model $MODEL"
    hf_time=$(eval $CMD | grep 'Average time' | awk '{print $6}')

    echo "$MODEL,$ds_time,$hf_time" >> results.csv
done



