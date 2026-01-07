# MODELNAME=LlaSMol-Mistral-7B-RLSF_try && CUDA_VISIBLE_DEVICES=0 accelerate launch rlsf_ppo_gen_try.py --model_path checkpoint/molecule_generation/LlaSMol-Mistral-7B-SFT_mg0.2.0/final_checkpoint --data_path piyush-j/rlsf_molecule_generation --base_model mistralai/Mistral-7B-v0.1 --wandb_run_name $MODELNAME --output_dir rlsf_checkpoint/try --num_epochs 1 --batch_size_multiplier 1 --mini_batch_size 1 --gradient_accumulation_steps 1

# MODELNAME=LlaSMol-Mistral-7B-RLSF_try && CUDA_VISIBLE_DEVICES=0 accelerate launch rlsf_ppo_gen_try.py --data_path piyush-j/rlsf_molecule_generation --base_model google/gemma-2b-it --wandb_run_name $MODELNAME --output_dir rlsf_checkpoint/try --num_epochs 1 --batch_size_multiplier 1 --mini_batch_size 1 --gradient_accumulation_steps 1

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
parser.add_argument('--model_path', type=str, required=False)
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

model = PeftModelForCausalLM.from_pretrained(
    model,
    args.model_path,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
)

model = model.merge_and_unload() # VERY IMPORTANT!!! Else the model after AutoModelForCausalLMWithValueHead produces gibberish

print("saving")

model.save_pretrained("try_full_model_save")

print("loading")

model = AutoModelForCausalLM.from_pretrained("try_full_model_save",
                # torch_dtype=torch.bfloat16,
                quantization_config = bnb_config,
                device_map={'':device_string})

print(model)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLMWithValueHead.from_pretrained(model,
                                            # torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             device_map={"":device_string},
                                             peft_config=lora_config)

print(model)

fromactualrun = "<s>[INST] Give me a molecule that satisfies the conditions outlined in the description: The molecule is an organonitrogen compound and an organooxygen compound. [/INST]"

# messages = [
#     {"role": "user", "content": fromactualrun},
# ]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

inputs = tokenizer(fromactualrun, return_tensors="pt")
inputs = inputs.to('cuda:0')

from transformers import GenerationConfig

generation_settings = {'num_return_sequences': 5, 'num_beams': 8}

generation_config = GenerationConfig(
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            **generation_settings,
        )

generation_output = model.generate(input_ids=inputs['input_ids'], max_new_tokens=100, generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,)

s = generation_output.sequences

output = tokenizer.batch_decode(s, skip_special_tokens=False)

print(output)

# "<bos><start_of_turn>user\nGive me a molecule that satisfies the conditions outlined in the description: The molecule is an organonitrogen compound and an organooxygen compound.<end_of_turn>
# Sure, here's a molecule that satisfies the conditions:\n\nCH3CH(NO2)CH(CH3)NO2\n\nThis molecule is an organonitrogen compound because it contains a nitrogen atom bonded to an organic carbon atom.<eos>"