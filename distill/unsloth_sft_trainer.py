from unsloth import FastLanguageModel, is_bfloat16_supported, get_chat_template, train_on_responses_only
from typing import Optional, Dict, Any, List
import random, torch, numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EvalPrediction, TrainerCallback
from datetime import datetime
from run_evaluation import get_response_references, load_jsonl, measure_verdict_matching_stats, measure_template_following_accuracy
from trl import SFTTrainer
import fire
from tqdm import tqdm
from transformers.trainer_callback import TrainerControl, TrainerState, TrainingArguments
import wandb


def load_llama_model_and_tokenizer(model_name, max_seq_length=512, load_in_4bit=True, dtype=None, cache_dir=None,
                                   lora_rank=16, lora_alpha=16, lora_dropout=0.01, seed=4311):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        cache_dir=cache_dir,
    )

    if 'llama-3.1' in model_name.lower():
        chat_template_name = 'llama-3.1'
    elif 'llama-3.2' in model_name.lower():
        chat_template_name = 'llama-3.2'
    elif 'llama-3.3' in model_name.lower():
        chat_template_name = 'llama-3.3'
    else:
        raise NotImplementedError

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template_name,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer

def load_gemma_model_and_tokenizer(model_name, max_seq_length=512, load_in_4bit=True, dtype=None, cache_dir=None,
                                   lora_rank=16, lora_alpha=16, lora_dropout=0.01, seed=4311):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        cache_dir=cache_dir,
    )

    if 'gemma-3' in model_name.lower():
        chat_template_name = 'gemma-3'
    else:
        raise NotImplementedError

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template_name,
    )

    model = FastLanguageModel.get_peft_model(
        model,

        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,

        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer

def load_model_and_tokenizer(model_name, max_seq_length=512, load_in_4bit=True, dtype=None, cache_dir=None,
                                   lora_rank=16, lora_alpha=16, lora_dropout=0.01, seed=4311):
    if 'llama' in model_name.lower():
        return load_llama_model_and_tokenizer(model_name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit,
                                                dtype=dtype, cache_dir=cache_dir,
                                                lora_rank=lora_rank, lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout, seed=seed)
    elif 'gemma' in model_name.lower():
        return load_gemma_model_and_tokenizer(model_name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit,
                                                dtype=dtype, cache_dir=cache_dir,
                                                lora_rank=lora_rank, lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout, seed=seed)
    else:
        raise NotImplementedError


def load_formatting_prompts_func(model_name, tokenizer):
    if 'llama' in model_name.lower():
        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                     convos]
            return {"text": texts, }
        return formatting_prompts_func
    elif 'gemma' in model_name.lower():
        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
                for convo in convos]
            return {"text": texts, }
        return formatting_prompts_func
    else:
        raise NotImplementedError



def run_trainer(
        # ─── data ────────────────────────────────────────────────────────────────
        train_data_file: str,
        eval_data_file: str,
        # ─── model / tokenizer ───────────────────────────────────────────────────
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length: int = 12000,
        load_in_4bit: bool = True,
        dtype: Optional[str] = None,
        cache_dir: str | None = None,
        # ─── optimization ────────────────────────────────────────────────────────
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        warmup_ratio: float = 0.01,
        lr_scheduler_type: str = "cosine_with_restarts",
        lora_alpha: int = 128,
        lora_rank: int = 64,
        eval_accumulation_steps: int = 1,
        save_total_limit: int = 1,
        lora_dropout: float = 0.0,
        # ─── bookkeeping ─────────────────────────────────────────────────────────
        logging_steps: int = 2,
        eval_steps: int = 2,
        custom_metrics_eval_steps: int = 1,
        save_steps: int = 2,
        output_dir: str = "./outputs/",
        seed: int = 42,
        # ─── custom‑metric settings ──────────────────────────────────────────────
        metric_n_samples: int = 64,
        max_gen_new_tokens: int = 1024,
        report_to: str = "wandb",
        generation_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Train an SFT model with every hyper‑parameter exposed as a function argument.
    A simple custom metric (`keyword_hit_rate`) is computed on the eval set:
        proportion of generated replies that contain `target_keyword`.
    Replace this logic with any generation‑based metric you like.
    """
    # ── seeding ──
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model, tokenizer = load_model_and_tokenizer(model_name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit,
                                                dtype=dtype, cache_dir=cache_dir,
                                                lora_rank=lora_rank, lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout, seed=seed)


    formatting_prompts_func = load_formatting_prompts_func(model_name, tokenizer)

    train_dataset = load_dataset("json", data_files=train_data_file, split='train')
    train_ds = train_dataset.map(formatting_prompts_func, batched=True, )
    print("train dataset size:", len(train_ds))

    print("sample of training input:", train_ds[0]['text'])

    eval_dataset = load_dataset("json", data_files=eval_data_file, split='train')
    eval_ds = eval_dataset.map(formatting_prompts_func, batched=True, )
    print("eval dataset size:", len(eval_ds))

    raw_eval_data = load_jsonl(eval_data_file)

    #todo: for test:
    # train_ds = train_ds.select(range(16))
    eval_ds = eval_ds.shuffle().select(range(100))

    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    class EvaluationCallback(TrainerCallback):
        def __init__(self, dataset, every, model, tokenizer, max_eval_samples=20):
            self.ds, self.every = dataset, every
            self.model = model
            self.tokenizer = tokenizer
            self.max_eval_samples = max_eval_samples

        def on_evaluate(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl, **kwargs):

            if state.global_step % self.every or not state.is_local_process_zero:
                return control

            random.shuffle(self.ds)
            sample_ds = self.ds[:self.max_eval_samples]

            model_cls = self.model
            tokenizer_cls = self.tokenizer
            inference_batch_size = 1
            responses = []
            references = []

            for batch_idx in tqdm(range(0, len(sample_ds), inference_batch_size)):
                batch = sample_ds[batch_idx:batch_idx + inference_batch_size]
                batch_responses, batch_references = get_response_references(batch, model_cls, tokenizer_cls)
                responses.extend(batch_responses)
                references.extend(batch_references)

            metrics = {}
            template_metrics = measure_template_following_accuracy(responses)
            verdict_metrics = measure_verdict_matching_stats(responses, references)
            metrics.update(template_metrics)
            metrics.update(verdict_metrics)

            wandb.log({"epoch": state.epoch,
                       "step": state.global_step,
                       **metrics})

            return control


    # ── training args ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_accumulation_steps=eval_accumulation_steps,
        save_total_limit=save_total_limit,
        eval_steps=eval_steps,
        run_name=run_name,
        save_steps=save_steps,
        bf16=is_bfloat16_supported(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=seed,
        report_to=report_to,
    )

    # ── custom metric ──
    if generation_kwargs is None:
        generation_kwargs = dict(max_new_tokens=max_gen_new_tokens, do_sample=False)


    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # callbacks=[EvaluationCallback(
        #     raw_eval_data, custom_metrics_eval_steps, model, tokenizer)],
        data_collator=DataCollatorForSeq2Seq(tokenizer, max_length=max_seq_length, return_tensors="pt"),
    )

    if 'llama' in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif 'gemma' in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    else:
        raise ValueError('unsupported model_name')

    for _ in range(2):
        sample = random.choice(trainer.train_dataset)
        sample_training_label = tokenizer.decode([x for x in sample["labels"] if x != -100 ])
        print("A sample of training label is:", sample_training_label)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(run_trainer)
