export MODEL='mistralai/Mistral-7B-Instruct-v0.2'
# export MODEL='mistralai/Mixtral-8x7B-Instruct-v0.1'
# export MODEL_PATH='/home/yc714/dataset/pretrained-models/Qwen2-7B-Instruct'
export PRED_2_PATH='./data_pipeline/corrections/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
export EVAL_PROMPT='qwen2-boxed-step'

export DATASET_PATH='./data/main_data.json'
export LEAKS_PATH='./data_pipeline/leaks/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.json'

PROMPT_TYPE='naive' # or 'privacy_enhanced'
CUDA_VISIBLE_DEVICES=0 python data_pipeline/get_final_action_with_leak_hints.py --main-path $DATASET_PATH --leaks-path $LEAKS_PATH --output-path $PRED_2_PATH"0.csv" --model $MODEL --prompt-type $PROMPT_TYPE --start-index 0 --num -1 &
# CUDA_VISIBLE_DEVICES=1 python eval_math.py --model $MODEL_PATH --remainder 1 --n_groups 8 --save_path $PRED_PATH"1.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=2 python eval_math.py --model $MODEL_PATH --remainder 2 --n_groups 8 --save_path $PRED_PATH"2.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=3 python eval_math.py --model $MODEL_PATH --remainder 3 --n_groups 8 --save_path $PRED_PATH"3.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=4 python eval_math.py --model $MODEL_PATH --remainder 4 --n_groups 8 --save_path $PRED_PATH"4.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=5 python eval_math.py --model $MODEL_PATH --remainder 5 --n_groups 8 --save_path $PRED_PATH"5.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=6 python eval_math.py --model $MODEL_PATH --remainder 6 --n_groups 8 --save_path $PRED_PATH"6.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=7 python eval_math.py --model $MODEL_PATH --remainder 7 --n_groups 8 --save_path $PRED_PATH"7.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1
