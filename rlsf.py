import argparse
import json
import os
import random
import time
from random import choices

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
# from google.colab import userdata
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModelForCausalLM
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline, set_seed)
from trl.trainer import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from accelerate import Accelerator, PartialState
from utils.general_prompter import GeneralPrompter, get_chat_content
from utils.chat_generation import generate_chat

from generation import extract_prediction_smiles
from rdkit import Chem

tqdm.pandas()

def set_random_seeds(seed: int = 12):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seeds()

os.environ["HF_TOKEN"] = None

os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--wandb_run_name', type=str, required=True)

parser.add_argument('--wandb_project', type=str, required=False, default='RLSF-Chem')
parser.add_argument('--wandb_log_model', type=bool, required=False, default=True)

parser.add_argument('--num_epochs', type=int, required=False, default=10)
parser.add_argument('--batch_size_multiplier', type=int, required=False, default=2)
parser.add_argument('--mini_batch_size', type=int, required=False, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=2)

parser.add_argument('--feedback_mode', type=str, required=False, choices=['binary', 'cert'], default='binary')
parser.add_argument('--penalty_value', type=float, required=False, default=0.0)
parser.add_argument('--rew_value', type=float, required=False, default=1.0)
args = parser.parse_args()
print(args)

os.environ["WANDB_LOG_MODEL"] = "true" if args.wandb_log_model else "false"

run = wandb.init(reinit=True,
                name = args.wandb_run_name,
                 project=args.wandb_project,
                 settings=wandb.Settings(start_method='fork'), 
                 save_code=True,
                 tags=[args.feedback_mode])

wandb.config.update(args)

accelerator = Accelerator()

# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(args.base_model, truncation=True, max_length=512, padding=False, token=os.environ['HF_TOKEN'])
tokenizer.sep_token = '<unk>'
tokenizer.cls_token = '<unk>'
tokenizer.mask_token = '<unk>'
# tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Load model

device_string = PartialState().process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
    bias="none",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(args.base_model,
                torch_dtype=torch.bfloat16,
                # quantization_config = bnb_config,
                device_map={'':device_string})

# model.load_adapter(args.model_path)
model = PeftModelForCausalLM.from_pretrained(
    model,
    args.model_path,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
)

model = model.merge_and_unload() # VERY IMPORTANT!!! Else the model after AutoModelForCausalLMWithValueHead produces gibberish

# print("saving merged model")

# model.save_pretrained(f"{args.output_dir}-full-sft-merged")

# print("loading again for AutoModelForCausalLMWithValueHead")

model = AutoModelForCausalLM.from_pretrained(f"{args.output_dir}-full-sft-merged",
                quantization_config = bnb_config,
                device_map={'':device_string})

model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLMWithValueHead.from_pretrained(model,
                                            # torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config, # this is ok
                                             device_map={"":device_string},
                                             peft_config=lora_config)

# following https://deci.ai/blog/how-to-instruction-tune-a-base-llm-using-qlora-with-decilm-6b/ and https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=Ybeyl20n3dYH
model.gradient_checkpointing_enable()

config = PPOConfig(
    batch_size=args.batch_size_multiplier*args.mini_batch_size*args.gradient_accumulation_steps, mini_batch_size=args.mini_batch_size, steps=10000, learning_rate=1.41e-5, remove_unused_columns=False, 
    log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps
) # remove_unused_columns False because we want to preserve the text version of query in the dataset

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataset = load_dataset(args.data_path, split="train")
print(dataset)

prompter = GeneralPrompter(get_chat_content, '[/INST]')

dataset = dataset.map(
    lambda x: {"real_input_text": prompter.generate_prompt(generate_chat(x['input']))}, # convert input to chat format
    batched=False,
)

dataset = dataset.map(
    lambda x: {"input_ids": tokenizer.encode(x["real_input_text"], return_tensors="pt")[0]},
    batched=False,
)

dataset.set_format("pytorch")

# print first 5 examples
for i in range(5):
    print(dataset[i])

"""### Initialize PPOTrainer
The `PPOTrainer` takes care of device placement and optimization later on:
"""

ppo_trainer = PPOTrainer(config, model, None, tokenizer, dataset=dataset, data_collator=collator)

"""### Generation settings
For the response generation we just use sampling and make sure top-k and nucleus sampling are turned off as well as a minimal length.

## Optimize model

### Training loop
"""

generation_kwargs = {
    "min_length": -1,
    "do_sample": True,
    "max_new_tokens": 256,
    "return_prompt": False,
    "temperature": 1,
    # "top_k": 0.0, # no top-k sampling
    # "top_p": 1.0, # no nucleus sampling
    "pad_token_id": tokenizer.eos_token_id,
}

def reward_function(response):
    extracted_prediction = extract_prediction_smiles(response)
    print("extracted_prediction: ", extracted_prediction)
    if not extracted_prediction: 
        return False
    return Chem.MolFromSmiles(extracted_prediction) # if mol is not None

step_counter = 0

for epoch in range(args.num_epochs):
    print("Epoch ", epoch)
    for batch in tqdm(ppo_trainer.dataloader):
        step_counter += 1
        torch.cuda.empty_cache()

        query_tensors = batch["input_ids"] # these are the tokenized list of tensors

        #### Get response from gemma
        response_tensors, ref_response_tensors = ppo_trainer.generate(query_tensors, generate_ref_response=True, **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

        #### Compute score
        rewards = []
        for input, real_input_text, response_i, ref_response_i in zip(batch["input"], batch["real_input_text"], batch["response"], batch["ref_response"]):
            print("input -> ", input)
            print("real_input_text -> ", real_input_text)
            print("response_i -> ", response_i)
            print("ref_response_i -> ", ref_response_i)
            if not reward_function(response_i):
                score_int = args.penalty_value
                print("incorrect response!")
            else:
                score_int = args.rew_value
                print("response is correct!")
            # reward_vector = reward_vector_generator()
            # reward_vector[0] = args.rew_value # ignore new line/<sos> mismatches
            # reward_vector[-1] = score_int
            if args.feedback_mode == 'binary':
                rewards.append(torch.tensor(score_int, dtype=torch.float16))
            else:
                raise NotImplementedError("cert mode not implemented")
                # rewards.append(torch.FloatTensor(reward_vector))
        
        if args.feedback_mode == 'binary':
            rewards_to_train = rewards
            rewards_to_log = rewards
        else:
            raise NotImplementedError("cert mode not implemented")
            # rewards_to_train = rewards
            # rewards_to_log = [r.mean() for r in rewards]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_to_train)
        ppo_trainer.log_stats(stats, batch, rewards_to_log, columns_to_log=["input", "real_input_text", "response", "ref_response"])

        if step_counter % 100 == 0:
            if not os.path.exists(f"{args.output_dir}-step{step_counter}"):
                os.mkdir(f"{args.output_dir}-step{step_counter}")
            ppo_trainer.model.save_pretrained(f"{args.output_dir}-step{step_counter}")
            print(f"Model saved: {args.output_dir}-step{step_counter}")
