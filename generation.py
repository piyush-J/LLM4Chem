import time
import pandas as pd
import torch
from transformers import GenerationConfig

from utils.chat_generation import generate_chat, generate_chat_v2
from utils.general_prompter import GeneralPrompter, get_chat_content
from utils.smiles_canonicalization import canonicalize_molecule_smiles

from model import load_tokenizer_and_model

from rdkit import Chem
import sys
from io import StringIO
from rdkit import rdBase
from rdkit import RDLogger

rdBase.WrapLogs()

def extract_prediction_smiles(output_text): #TODO (P3): generalize this function to extract any tag
    left_tag = '<SMILES>'
    right_tag = '</SMILES>'
    assert isinstance(output_text, str)
    left_tag_pos = output_text.find(left_tag)
    right_tag_pos = output_text.find(right_tag)
    if left_tag_pos == -1 or right_tag_pos == -1:
        return ''
    smiles = output_text[left_tag_pos + len(left_tag) : right_tag_pos].strip()
    return smiles

def rdkit_eval_function(list_of_convo):
    sio = sys.stderr = StringIO()
    RDLogger.EnableLog('rdApp.*') # enable the error logger, because we need it to catch the error message
    mol = Chem.MolFromSmiles(extract_prediction_smiles(list_of_convo[-1]))
    print("m:", mol)
    feedback = sio.getvalue().strip()
    if mol is None: # error
        print("error message:", feedback, "<end>")
        correctness_string = f'{feedback}. This is wrong. Using this feedback, please try again. Based on the reactants and reagents given above, suggest a possible product.'
    else: 
        correctness_string = None
    # sio = sys.stderr = StringIO() # reset the error logger
    sys.stderr = sys.__stderr__
    RDLogger.DisableLog('rdApp.*') # back to default
    return correctness_string

def tokenize(tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    result = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def canonicalize_smiles_in_text(text, tags=('<SMILES>', '</SMILES>'), keep_text_unchanged_if_no_tags=True, keep_text_unchanged_if_error=False):
    try:
        left_tag, right_tag = tags
        assert left_tag is not None
        assert right_tag is not None
        
        left_tag_pos = text.find(left_tag)
        right_tag_pos = None
        if left_tag_pos == -1:
            assert right_tag not in text, 'The input text "%s" only contains the right tag "%s" but no left tag"%s"' % (text, right_tag, left_tag)
            return text
        else:
            right_tag_pos = text.find(right_tag)
            assert right_tag_pos is not None, 'The input text "%s" only contains the left tag "%s" but no right tag"%s"' % (text, left_tag, right_tag)
    except AssertionError:
        if keep_text_unchanged_if_no_tags:
            return text
        raise
    
    smiles = text[left_tag_pos + len(left_tag) : right_tag_pos].strip()
    try:
        smiles = canonicalize_molecule_smiles(smiles, return_none_for_error=False)
    except KeyboardInterrupt:
        raise
    except:
        if keep_text_unchanged_if_error:
            return text
        raise

    new_text = text[:left_tag_pos] + ('' if (left_tag_pos == 0 or text[left_tag_pos - 1] == ' ') else ' ') + left_tag + ' ' + smiles + ' ' + right_tag + ' ' + text[right_tag_pos + len(right_tag):].lstrip()
    return new_text


class LlaSMolGeneration(object):
    def __init__(self, model_name, base_model=None, device=None):
        self.prompter = GeneralPrompter(get_chat_content)

        self.tokenizer, self.model = load_tokenizer_and_model(model_name, base_model=base_model, device=device)
        self.device = self.model.device  # TODO: check if this can work
        self.csvlogger = pd.DataFrame(columns=['input_original', 'feedback_iteration_no', 'input_current_entire_conversation', 'output', 'feedback', 'input_too_long'])

    def create_sample(self, text, canonicalize_smiles=True, max_input_tokens=None):
        if canonicalize_smiles:
            real_text = canonicalize_smiles_in_text(text)
        else:
            real_text = text
        
        sample = {'input_text': text}
        chat = generate_chat_v2(real_text, output_text=None)
        print("chat: ", chat)
        full_prompt = self.prompter.generate_prompt(chat)
        print("full_prompt: ", full_prompt)
        sample['real_input_text'] = full_prompt
        tokenized_full_prompt = tokenize(self.tokenizer, full_prompt, add_eos_token=False)
        sample.update(tokenized_full_prompt)
        if max_input_tokens is not None and len(tokenized_full_prompt['input_ids']) > max_input_tokens:
            sample['input_too_long'] = True
        
        return sample
    
    def _generate(self, input_ids, max_new_tokens=1024, **generation_settings):
        generation_config = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **generation_settings,
        )
        self.model.eval()
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=False)
        output_text = []
        for output_item in output:
            text = self.prompter.get_response(output_item)
            output_text.append(text)

        return output_text, output

    def generate_with_feedback(self, input_text, batch_size=1, max_input_tokens=512, max_new_tokens=1024, canonicalize_smiles=True, print_out=False, **generation_settings):
        # This part is called from generate_on_dataset.py where input_text is a list of one string (as the batch size is 1)
        
        feedback_limit = 3
        list_of_convo = []

        if isinstance(input_text, str):
            input_text = [input_text]
        else:
            input_text = list(input_text)
        assert len(input_text) == 1

        samples = []
        for text in input_text:
            list_of_convo.append(text)
            sample = self.create_sample(text, canonicalize_smiles=canonicalize_smiles, max_input_tokens=max_input_tokens)
            samples.append(sample)
        
        all_outputs = []
        k = 0
        while True:
            if k >= len(samples):
                break
            e = min(k + batch_size, len(samples))

            batch_samples = []
            skipped_samples = []
            batch_outputs = []
            original_index = {}
            
            for bidx, sample in enumerate(samples[k: e]):
                if 'input_too_long' in sample and sample['input_too_long']:
                    original_index[bidx] = ('s', len(skipped_samples))
                    skipped_samples.append(sample)
                    continue
                original_index[bidx] = ('b', len(batch_samples))
                batch_samples.append(sample)

            if len(batch_samples) > 0: 
                input_ids = {'input_ids': [sample['input_ids'] for sample in batch_samples]}
                feedback_counter = 1
                while feedback_counter <= feedback_limit:
                    input_ids = {'input_ids': [sample['input_ids']]} # asserted that len(batch_samples) == 1
                    input_ids = self.tokenizer.pad(
                        input_ids,
                        padding=True,
                        return_tensors='pt'
                    ) # TODO (P3): check - I think this is not needed if batch size is 1
                    input_ids = input_ids['input_ids'].to(self.device)
                    batch_output_text, _ = self._generate(input_ids, max_new_tokens=max_new_tokens, **generation_settings)
                    # batch_output_text = ['This is a test output'+str(feedback_counter)] * 10 # dummy
                    output_text_for_eval = batch_output_text[0] # i.e., num_return_sequences = 1
                    list_of_convo.append(output_text_for_eval)
                    
                    feedback = rdkit_eval_function(list_of_convo) # feedback is being called at the end of the loop too - log to evaluate the correctness of the final output

                    # batch_samples[0] - assuming that batch size is 1
                    self.csvlogger = self.csvlogger._append({'input_original': input_text[0], 
                                            'feedback_iteration_no': feedback_counter, 
                                            'input_current_entire_conversation': batch_samples[0]['real_input_text'], 
                                            'output': output_text_for_eval, 
                                            'feedback': feedback,
                                            'input_too_long': True if 'input_too_long' in batch_samples[0] and batch_samples[0]['input_too_long'] else False}, ignore_index=True)
                    
                    if feedback is None: # the answer is correct
                        break 

                    list_of_convo.append(feedback)

                    mod_sample = self.create_sample(list_of_convo, canonicalize_smiles=False, max_input_tokens=max_input_tokens)
                    batch_samples = [mod_sample] # changing the sample to the modified sample for the next iteration

                    if 'input_too_long' in mod_sample and mod_sample['input_too_long']:
                        # warn user
                        raise Warning(f'Modified input text exceeds the token limit ({max_input_tokens})')
                    
                    feedback_counter += 1
            
                batch_outputs = [[output_text_for_eval]] # the last one after the last iteration of feedback

            new_batch_samples = []
            new_batch_outputs = []

            for bidx in sorted(original_index.keys()):
                place, widx = original_index[bidx]
                if place == 'b':
                    sample = batch_samples[widx]
                    output = batch_outputs[widx]
                elif place == 's':
                    sample = skipped_samples[widx]
                    output = None
                else:
                    raise ValueError(place)
                new_batch_samples.append(sample)
                new_batch_outputs.append(output)

            batch_samples = new_batch_samples
            batch_outputs = new_batch_outputs

            assert len(batch_samples) == len(batch_outputs)
            for sample, sample_outputs in zip(batch_samples, batch_outputs):
                if print_out:
                    print('=============')
                    print('Input: %s' % sample['input_text'])
                    if sample_outputs is None:
                        print('Output: None (Because the input text exceeds the token limit (%d) )' % max_input_tokens)
                    else:
                        for idx, output_text in enumerate(sample_outputs, start=1):
                            print('Output %d: %s' % (idx, output_text))
                    print('\n')

                log = {
                    'input_text': sample['input_text'], 
                    'real_input_text': sample['real_input_text'],
                    'output': sample_outputs,
                }

                all_outputs.append(log)

            print("all_outputs: ", all_outputs) # NOTE: the input_text and real_input_text is the one after generating the output and not the original input_text and real_input_text. We can use this to continue the conversation.
            # all_outputs = [{'input_text': '<SMILES> CCOC(=N)C1=C(F)C=CC=C1F.O=C=NC1=CC(Cl)=C(OC2=NC=C(C(F)(F)F)C=C2Cl)C(Cl)=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product.', 'real_input_text': '<s>[INST] <SMILES> CCOC(=N)C1=C(F)C=CC=C1F.O=C=NC1=CC(Cl)=C(OC2=NC=C(C(F)(F)F)C=C2Cl)C(Cl)=C1 </SMILES> Based on the reactants and reagents given above, suggest a possible product. [/INST]', 'output': ['This is a test output']}]

            k = e
        
        return all_outputs

    def generate(self, input_text, batch_size=1, max_input_tokens=512, max_new_tokens=1024, canonicalize_smiles=True, print_out=False, **generation_settings):
            if isinstance(input_text, str):
                input_text = [input_text]
            else:
                input_text = list(input_text)
            assert len(input_text) > 0

            samples = []
            for text in input_text:
                sample = self.create_sample(text, canonicalize_smiles=canonicalize_smiles, max_input_tokens=max_input_tokens)
                samples.append(sample)
            
            all_outputs = []
            k = 0
            while True:
                if k >= len(samples):
                    break
                e = min(k + batch_size, len(samples))

                batch_samples = []
                skipped_samples = []
                batch_outputs = []
                original_index = {}
                
                for bidx, sample in enumerate(samples[k: e]):
                    if 'input_too_long' in sample and sample['input_too_long']:
                        original_index[bidx] = ('s', len(skipped_samples))
                        skipped_samples.append(sample)
                        continue
                    original_index[bidx] = ('b', len(batch_samples))
                    batch_samples.append(sample)

                if len(batch_samples) > 0:
                    input_ids = {'input_ids': [sample['input_ids'] for sample in batch_samples]}
                    input_ids = self.tokenizer.pad(
                        input_ids,
                        padding=True,
                        return_tensors='pt'
                    )
                    input_ids = input_ids['input_ids'].to(self.device)
                    batch_output_text, _ = self._generate(input_ids, max_new_tokens=max_new_tokens, **generation_settings)
                    num_batch_samples = len(batch_samples)
                    ko = 0
                    num_return_sequences = 1 if 'num_return_sequences' not in generation_settings else generation_settings['num_return_sequences']
                    for sample in range(num_batch_samples):
                        sample_outputs = []
                        for _ in range(num_return_sequences):
                            sample_outputs.append(batch_output_text[ko])
                            ko += 1
                        batch_outputs.append(sample_outputs)
                    
                new_batch_samples = []
                new_batch_outputs = []

                for bidx in sorted(original_index.keys()):
                    place, widx = original_index[bidx]
                    if place == 'b':
                        sample = batch_samples[widx]
                        output = batch_outputs[widx]
                    elif place == 's':
                        sample = skipped_samples[widx]
                        output = None
                    else:
                        raise ValueError(place)
                    new_batch_samples.append(sample)
                    new_batch_outputs.append(output)

                batch_samples = new_batch_samples
                batch_outputs = new_batch_outputs

                assert len(batch_samples) == len(batch_outputs)
                for sample, sample_outputs in zip(batch_samples, batch_outputs):
                    if print_out:
                        print('=============')
                        print('Input: %s' % sample['input_text'])
                        if sample_outputs is None:
                            print('Output: None (Because the input text exceeds the token limit (%d) )' % max_input_tokens)
                        else:
                            for idx, output_text in enumerate(sample_outputs, start=1):
                                print('Output %d: %s' % (idx, output_text))
                        print('\n')

                    log = {
                        'input_text': sample['input_text'], 
                        'real_input_text': sample['real_input_text'],
                        'output': sample_outputs,
                    }

                    all_outputs.append(log)

                k = e
            
            return all_outputs