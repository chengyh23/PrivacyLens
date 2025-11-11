# train_dpo.py
import logging
import sys
from dataclasses import dataclass, field

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

def apply_chat_template(example, tokenizer, task, prompt):
    # TODO tokenizer.apply_chat_template
    assert task in ["dpo"]

    if prompt == "qwen2-boxed":
        prompt_no_input = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        raise NotImplementedError(f"Prompt {prompt} is not supported currently")

    text_chosen = example['chosen']
    text_rejected = example['rejected']
    if prompt == 'qwen2-boxed':
        new_example = {
            'prompt': prompt_no_input.format(instruction=example['prompt']),
            'chosen': text_chosen,
            'rejected': text_rejected,
        }
    else:
        raise NotImplementedError(f"Prompt {prompt} is not supported currently")
    return new_example


@dataclass
class StepDPOConfig(DPOConfig):
    data_path: str = field(default="data_pipeline/dataset.json")
    prompt: str = field(default="alpaca")

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
    ref_model = model
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    # ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model=model, 
        ref_model=ref_model,
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets.keys() else None,
    )
    
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