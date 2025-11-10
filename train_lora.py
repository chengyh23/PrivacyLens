# train_dpo.py
import logging
import sys
from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model

logger = logging.getLogger(__name__)

from train import apply_chat_template

@dataclass
class StepDPOConfig(DPOConfig):
    data_path: str = field(default="data_pipeline/dataset.json")
    prompt: str = field(default="alpaca")

def disable_model_cache(model):
    """Recursively disable use_cache in model config and layers."""
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    # Also disable cache in transformer layers if they have the attribute
    if hasattr(model, 'modules'):
        for module in model.modules():
            if hasattr(module, 'config') and hasattr(module.config, 'use_cache'):
                module.config.use_cache = False

def merge_lora():
    # 1. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "base_model_name", torch_dtype=torch.float16, device_map="auto"
    )

    # 2. Load the PEFT model with adapter
    peft_model = PeftModel.from_pretrained(
        base_model, "path/to/adapter", torch_dtype=torch.float16
    )

    # 3. Merge adapter weights with base model
    merged_model = peft_model.merge_and_unload()

def main():    
    parser = TrlParser((ModelConfig, StepDPOConfig))
    model_args, training_args = parser.parse_args_and_config()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()

    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.info(f"Model parameters {model_args}")
    # logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed for reproducibility
    set_seed(training_args.seed)


    ###############
    # Load datasets
    ###############
    # train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    if ".json" in training_args.data_path:
        raw_datasets = load_dataset(
            "json",
            data_files=training_args.data_path.split("||"),
        )
    else:
        raw_datasets = load_dataset(training_args.data_path)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    
    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo", "prompt": training_args.prompt},
        num_proc=training_args.dataloader_num_workers,
        # num_proc=1,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )


    model = model_args.model_name_or_path
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    #########################
    # Instantiate PEFT model
    #########################
    rank_dimension = 6
    lora_alpha = 8
    # lora_dropout = 0.05
    lora_dropout = 0.0
    peft_config = LoraConfig(
        r=rank_dimension,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # target_modules=[
        #     "q_proj", "k_proj", "v_proj",
        #     "o_proj", "gate_proj", "up_proj",
        #     "down_proj"
        #     ],
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
		model,
		low_cpu_mem_usage=True,
		torch_dtype=torch.bfloat16,
		load_in_4bit=True,
		# use_flash_attention_2=True,
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_quant_type="nf4",
	)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    
    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model=model, 
        # ref_model=ref_model,
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets.keys() else None,
        peft_config=peft_config,
    )
    
    # # Disable use_cache for gradient checkpointing compatibility
    # # This is critical to avoid CheckpointError with gradient checkpointing
    # if training_args.gradient_checkpointing:
    #     try:
    #         # Get the actual model (may be wrapped by accelerate/deepspeed)
    #         model_to_fix = trainer.model
    #         if hasattr(trainer.model, 'module'):
    #             # Model is wrapped by DataParallel or similar
    #             model_to_fix = trainer.model.module
            
    #         if isinstance(model_to_fix, PeftModel):
    #             # For PEFT models, access the base model config
    #             try:
    #                 base_model = model_to_fix.get_base_model()
    #                 disable_model_cache(base_model)
    #                 logger.info("Disabled use_cache for PEFT base model to support gradient checkpointing")
    #             except Exception as e:
    #                 logger.warning(f"Could not access base model directly: {e}. Setting on PEFT wrapper.")
    #                 if hasattr(model_to_fix, 'config'):
    #                     model_to_fix.config.use_cache = False
    #         else:
    #             disable_model_cache(model_to_fix)
    #             logger.info("Disabled use_cache for model to support gradient checkpointing")
            
    #         # Also disable on the wrapper config if it exists (for PEFT models)
    #         if hasattr(trainer.model, 'config'):
    #             trainer.model.config.use_cache = False
            
    #         # Handle ref_model if it exists (may be None with PEFT)
    #         if hasattr(trainer, 'ref_model') and trainer.ref_model is not None:
    #             ref_model_to_fix = trainer.ref_model
    #             if hasattr(trainer.ref_model, 'module'):
    #                 ref_model_to_fix = trainer.ref_model.module
                
    #             if isinstance(ref_model_to_fix, PeftModel):
    #                 try:
    #                     ref_base_model = ref_model_to_fix.get_base_model()
    #                     disable_model_cache(ref_base_model)
    #                 except Exception as e:
    #                     logger.warning(f"Could not access ref base model directly: {e}")
    #                     if hasattr(ref_model_to_fix, 'config'):
    #                         ref_model_to_fix.config.use_cache = False
    #             else:
    #                 disable_model_cache(ref_model_to_fix)
    #             logger.info("Disabled use_cache for ref_model to support gradient checkpointing")
    #     except Exception as e:
    #         logger.error(f"Error disabling use_cache: {e}. This may cause CheckpointError.")
    #         raise
    
    ###############
    # Training loop
    ###############
    checkpoint = None

    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        else:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Resuming training from checkpoint at: {last_checkpoint}")
                checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(raw_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Training complete ***")

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {  # https://github.com/huggingface/trl/blob/main/trl/trainer/base_trainer.py#L27
            "model_name": training_args.hub_model_id,
            "dataset_name": training_args.data_path,
            "tags": ["alignment-handbook"],
        }
        # kwargs = {
        #     "finetuned_from": model_args.model_name_or_path,
        #     "dataset": [training_args.data_path],
        #     "dataset_tags": [training_args.data_path],
        #     "tags": ["alignment-handbook"],
        # }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")

if __name__ == "__main__":
    main()