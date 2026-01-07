import os
import random
from typing import List

import fire
import numpy as np
import torch
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, logging, set_seed)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.chat_generation import generate_chat
from utils.core_tagger import CoreTagger
from utils.general_prompter import GeneralPrompter, get_chat_content, get_chat_content_galactica

from extract_prediction import extract_answer_part
from config import TASKS, TASKS_WITH_SEMICOLON_REPLACE, TASKS_WITH_READING_GOLD_FROM_DATASET, TASK_TAGS
from utils.metrics import calculate_smiles_metrics

import wandb

def set_random_seeds(seed: int = 12):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seeds()

def train(
    base_model: str = "", 
    data_path: str = "",
    task: str = "",
    output_dir: str = "checkpoint",
    # training hyperparams
    batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    optim="adamw_bnb_8bit",
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    modules_to_save: List[str] = [],
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "", 
    wandb_log_model: str = "", 

    train_split='train',
    dev_split='validation',

    gradient_accumulation_steps: int = 2,
    max_seq_length: int = 512,
    lr_scheduler_type: str = "cosine",
    num_warmup_steps: int = 1000,
    weight_decay: float = 0.05,
    log_freq: int = 1,
    eval_freq: int = 100,
    save_freq: int = 100,
    remove_unused_columns: bool = True,
    gradient_checkpointing: bool = False,
    compute_metrics: bool = True
    ):

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    
    # if use_wandb:
    #     run = wandb.init(reinit=True, 
    #                     project=wandb_project, 
    #                     settings=wandb.Settings(start_method='fork'), 
    #                     save_code=True)

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    os.makedirs(output_dir, exist_ok=True)
    logging.set_verbosity_error()

    accelerator = Accelerator()

    # LOAD TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(base_model, truncation=True, max_length=max_seq_length, padding=False)
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"

    # LOAD EVALUATION METRICS

    read_gold_from_dataset=True if task in TASKS_WITH_READING_GOLD_FROM_DATASET else False
    replace_semicolon = True if task in TASKS_WITH_SEMICOLON_REPLACE else False

    split_set = load_dataset(data_path, split=dev_split)
    split_set_dict = split_set.to_dict()
    split_set_dict = [dict(zip(split_set_dict, t)) for t in zip(*split_set_dict.values())]
    split_set_dict[0]

    input_to_gold = None
    if read_gold_from_dataset: # Read gold from dataset for tasks that have multiple gold answers for one input
        input_to_gold = dict()
        for sample in split_set:
            input_key = sample['raw_input']
            if 'target' in sample and sample['target'] is not None:
                input_key = (input_key, sample['target'])
            gold_answer = sample['raw_output']
            if input_key not in input_to_gold: # For some tasks, there are multiple gold answers for one input
                input_to_gold[input_key] = []
            input_to_gold[input_key].append(gold_answer)
        
        for input_key in input_to_gold:
            input_to_gold[input_key] = set(input_to_gold[input_key])

    def read_result_eval(extracted_answers, task, replace_semicolon=False, read_gold_from_dataset=False):
        pred_list = [] # List of list of predictions
        gold_list = [] # List of list of gold answers

        for (item, ans) in zip(split_set_dict, extracted_answers):
            item['pred'] = [ans]
            item['gold'] = item['raw_output']
            item['input'] = item['raw_input']
            if read_gold_from_dataset:
                input_key = item['input']
                if 'target' in item and item['target'] is not None:
                    input_key = (input_key, item['target'])
                golds = input_to_gold[input_key]
                assert item['gold'] in golds
                if replace_semicolon:
                    new_golds = []
                    for one_gold in golds:
                        one_gold = one_gold.replace(';', '.')
                        new_golds.append(one_gold)
                    golds = new_golds
                else:
                    golds = list(golds)
            else:
                gold = item['gold']
                if replace_semicolon:
                    gold = gold.replace(';', '.')
                golds = [gold]
            gold_list.append(golds)

            preds = item['pred']
            if preds is None:  # Input too long, so skipped this sample
                pred_list.append(preds)
                continue

            new_preds = []
            for pred in preds:
                if replace_semicolon and pred is not None:
                    pred = pred.replace(';', '.')
                new_preds.append(pred)
            pred_list.append(new_preds)
        return pred_list, gold_list

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def compute_val_metrics(pred_label):
        print("Computing metrics")
        pred_ids = pred_label.predictions
        # labels_ids = pred_label.label_ids

        pred_ids[pred_ids == -100] = tokenizer.pad_token_id

        outputs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        extracted_answers = extract_answer_part(outputs, *(TASK_TAGS[task]), mode='tag')

        pred_list, gold_list = read_result_eval(extracted_answers, task, replace_semicolon, read_gold_from_dataset)

        metrics_output = calculate_smiles_metrics(pred_list, gold_list)
        return metrics_output 

    # LOAD PROMPTER

    prefix_chat = None

    if 'mistral' in base_model.lower():
        apply_chat_template_func = get_chat_content
        response_split = '[/INST]'
    elif 'galactica' in base_model.lower():
        apply_chat_template_func = get_chat_content_galactica
        response_split = '[START_REF]'
    elif 'gemma' in base_model.lower():
        apply_chat_template_func = tokenizer.apply_chat_template
        response_split = '<end_of_turn>'
    else:
        raise NotImplementedError

    prompter = GeneralPrompter(apply_chat_template_func, response_split)

    core_tagger = CoreTagger(tokenizer, core_tags_as_special_tokens=False, include_tags=True)

    # LOAD DATASET

    train_dataset = load_dataset(data_path, split=train_split)
    train_dataset = train_dataset.shuffle().map(lambda x: {'messages': generate_chat(x['input'], x['output'])})

    eval_dataset = load_dataset(data_path, split=dev_split)
    eval_dataset = eval_dataset.map(lambda x: {'messages': generate_chat(x['input'], x['output'])})

    def formatting_prompts_func(examples):
        output_texts = []
        for msg in examples['messages']:
            text = prompter.generate_prompt(msg)
            output_texts.append(text)
        return output_texts
    
    collator = DataCollatorForCompletionOnlyLM(response_split, tokenizer=tokenizer) # training on completion only

    # LOAD MODEL

    print("Loading the model")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_string = PartialState().process_index

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={'':device_string}
    )

    # model = get_peft_model(model, lora_config) # sft does that for you

    print("Starting main loop")

    sft_config = SFTConfig(
        output_dir = output_dir,
        optim=optim,
        dataloader_drop_last = False,
        evaluation_strategy = "steps",
        num_train_epochs = num_epochs,
        eval_steps = eval_freq,
        save_strategy = "steps",
        save_steps = save_freq,
        save_only_model = True,
        logging_steps = log_freq,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = 1,
        learning_rate = learning_rate,
        lr_scheduler_type = lr_scheduler_type,
        warmup_steps = num_warmup_steps,
        gradient_accumulation_steps = gradient_accumulation_steps,
        gradient_checkpointing = gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False} if gradient_checkpointing else None,
        weight_decay = weight_decay,
        ddp_find_unused_parameters = False,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        remove_unused_columns=remove_unused_columns, # Bug in SFTtrainer: setting it to True will solve the tensor str error
    )

    trainer = accelerator.prepare(SFTTrainer(
        model = model,
        args = sft_config,
        max_seq_length = max_seq_length,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        peft_config = lora_config,
        tokenizer = tokenizer,
        packing = False,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics = compute_val_metrics if compute_metrics else None, 
        preprocess_logits_for_metrics = preprocess_logits_for_metrics if compute_metrics else None
    ))

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.save_model(os.path.join(output_dir, "final_checkpoint/"))
    
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
