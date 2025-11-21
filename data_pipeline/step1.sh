export MODEL='mistralai/Mistral-7B-Instruct-v0.2'
# export MODEL='mistralai/Mixtral-8x7B-Instruct-v0.1'
# MODEL='Qwen/Qwen2-7B-Instruct'
# export MODEL_PATH='/home/yc714/dataset/pretrained-models/Qwen2-7B-Instruct'
export PRED_PATH='./data_pipeline/predictions/Mistral-7B-Instruct-v0.2'

export EVAL_PROMPT='qwen2-boxed-step'

export DATASET_PATH='./data/main_data.json'
PROMPT_TYPE='naive' # or 'privacy_enhanced'
CUDA_VISIBLE_DEVICES=2 python evaluation/get_final_action.py --input-path $DATASET_PATH --output-path $PRED_PATH".csv" --model $MODEL --prompt-type $PROMPT_TYPE --start-index 0 --num 10 --gpu-memory-utilization 0.5 &
# CUDA_VISIBLE_DEVICES=1 python eval_math.py --model $MODEL_PATH --remainder 1 --n_groups 8 --save_path $PRED_PATH"1.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=2 python eval_math.py --model $MODEL_PATH --remainder 2 --n_groups 8 --save_path $PRED_PATH"2.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=3 python eval_math.py --model $MODEL_PATH --remainder 3 --n_groups 8 --save_path $PRED_PATH"3.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=4 python eval_math.py --model $MODEL_PATH --remainder 4 --n_groups 8 --save_path $PRED_PATH"4.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=5 python eval_math.py --model $MODEL_PATH --remainder 5 --n_groups 8 --save_path $PRED_PATH"5.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=6 python eval_math.py --model $MODEL_PATH --remainder 6 --n_groups 8 --save_path $PRED_PATH"6.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=7 python eval_math.py --model $MODEL_PATH --remainder 7 --n_groups 8 --save_path $PRED_PATH"7.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1
