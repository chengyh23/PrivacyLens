ACCELERATE_LOG_LEVEL=info accelerate launch --gpu_ids 1,2 --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
    --num_processes 2 \
    train.py \
    --config configs/config_full.yaml \
    --model_name_or_path="Qwen/Qwen2-0.5B-Instruct" \
    --data_path="data_pipeline/dataset.json" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.4 \
    --num_train_epochs=4 \
    --save_strategy='steps' \
    --save_steps=200 \
    --save_total_limit=1 \
    --output_dir=outputs/qwen2-0.5b-instruct-step-dpo \
    --hub_model_id=qwen2-0.b-instruct-step-dpo \
    --prompt=qwen2-boxed
    # --model_name_or_path="Qwen/Qwen2-72B-Instruct" \
    # --output_dir=outputs/qwen2-72b-instruct-step-dpo \
    # --hub_model_id=qwen2-72b-instruct-step-dpo \
# python train.py configs/config_full.yaml \