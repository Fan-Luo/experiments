
# coding: utf-8

# ### convert hotpotqa to squard format

# According to Longformer: use the following input format with special tokens:  “[CLS] [q] question [/q] [p] sent1,1 [s] sent1,2 [s] ... [p] sent2,1 [s] sent2,2 [s] ...” 
# where [s] and [p] are special tokens representing sentences and paragraphs. The special tokens were added to the RoBERTa vocabulary and randomly initialized before task finetuning.

# In[1]:


# helper functions to convert hotpotqa to squard format modified from  https://github.com/chiayewken/bert-qa/blob/master/run_hotpot.py

import tqdm

def create_example_dict(context, answers, id, is_impossible, question, is_sup_fact, is_supporting_para):
    return {
        "context": context,
        "qas": [                        # each context corresponds to only one qa in hotpotqa
            {
                "answers": answers,
                "id": id,
                "is_impossible": is_impossible,
                "question": question,
                "is_sup_fact": is_sup_fact,
                "is_supporting_para": is_supporting_para
            }
        ],
    }


def create_para_dict(example_dicts):
    if type(example_dicts) == dict:
        example_dicts = [example_dicts]   # each paragraph corresponds to only one [context, qas] in hotpotqa
    return {"paragraphs": example_dicts}   


# In[2]:


import re
def convert_hotpot_to_squad_format(json_dict, gold_paras_only=False):
    
    """function to convert hotpotqa to squard format.


    Note: A context corresponds to several qas in SQuard. In hotpotqa, one question corresponds to several paragraphs as context. 
          "paragraphs" means different: each paragraph in SQuard contains a context and a list of qas; while 10 paragraphs in hotpotqa concatenated into a context for one question.

    Args:
        json_dict: The original data load from hotpotqa file.
        gold_paras_only: when is true, only use the 2 paragraphs that contain the gold supporting facts; if false, use all the 10 paragraphs
 

    Returns:
        new_dict: The converted dict of hotpotqa dataset, use it as a dict would load from SQuAD json file
                  usage: input_data = new_dict["data"]   https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py#L230

    """

    new_dict = {"data": []} 
    for example in json_dict: 

        support_para = set(
            para_title for para_title, _ in example["supporting_facts"]
        )
        sp_set = set(list(map(tuple, example['supporting_facts'])))
        
        raw_contexts = example["context"]
        if gold_paras_only: 
            raw_contexts = [lst for lst in raw_contexts if lst[0] in support_para]
            
        contexts = [" <s> ".join(lst[1]) for lst in raw_contexts]    # extra space is fine, which would be ignored latter. most sentences has already have heading space, there are several no heading space 
        context = " <p> " + " <p> ".join(contexts)
        
        is_supporting_para = []  # a boolean list with 10 True/False elements, one for each paragraph
        is_sup_fact = []         # a boolean list with True/False elements, one for each context sentence
        for para_title, para_lines in raw_contexts:
            is_supporting_para.append(para_title in support_para)   
            for sent_id, sent in enumerate(para_lines):
                is_sup_fact.append( (para_title, sent_id) in sp_set )


        answer = example["answer"].strip() 
        if answer.lower() == 'yes':
            answers = [{"answer_start": -1, "answer_end": -1, "text": answer}] 
        elif answer.lower() == 'no':
            answers = [{"answer_start": -2, "answer_end": -2, "text": answer}] 
        else:
            answers = []          # keep all the occurences of answer in the context
            for m in re.finditer(re.escape(answer), context):    
                answer_start, answer_end = m.span() 
                answers.append({"answer_start": answer_start, "answer_end": answer_end, "text": answer})
        if(len(answers) > 0):     
            new_dict["data"].append(
                create_para_dict(
                    create_example_dict(
                        context=context,
                        answers=answers,
                        id = example["_id"],
                        is_impossible=(answers == []),
                        question=example["question"],
                        is_sup_fact = is_sup_fact,
                        is_supporting_para = is_supporting_para 
                    )
                )
            ) 
    return new_dict


# ### longfomer's fine-tuning
# 
# 
# - For answer span extraction we use BERT’s QA model with addition of a question type (yes/no/span) classification head over the first special token ([CLS]).
# 
# - For evidence extraction we apply 2 layer feedforward networks on top of the representations corresponding to sentence and paragraph tokens to get the corresponding evidence prediction scores and use binary cross entropy loss to train the model.
# 
# - We combine span, question classification, sentence, and paragraphs losses and train the model in a multitask way using linear combination of losses.
# 

# In[4]:


### Section2: This is modified from longfomer's fine-tuning with triviaqa.py from https://github.com/allenai/longformer/blob/master/scripts/triviaqa.py
# !conda install transformers --yes
# !conda install cudatoolkit=10.0 --yes
# !python -m pip install git+https://github.com/allenai/longformer.git
# !conda install -c conda-forge regex --force-reinstall --yes
# !conda install pytorch-lightning -c conda-forge
# !pip install jdc 
# !pip install test-tube 
# !conda install ipywidgets --yes
# !conda update --force conda --yes  
# !jupyter nbextension enable --py widgetsnbextension 
# !conda install jupyter --yes

# need to run this every time start this notebook, to add python3.7/site-packages to sys.pat, in order to import ipywidgets, which is used when RobertaTokenizer.from_pretrained('roberta-base') 
import sys
sys.path.insert(-1, '/xdisk/msurdeanu/fanluo/miniconda3/lib/python3.7/site-packages')

import os
from collections import defaultdict
import json
import string
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.logging import TestTubeLogger    # sometimes pytorch_lightning.loggers works instead

from longformer.longformer import Longformer, LongformerConfig 
from longformer.sliding_chunks import pad_to_window_size
from more_itertools import locate
from collections import Counter 

# #### class hotpotqaDataset

# In[7]:


class hotpotqaDataset(Dataset):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    
    
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = convert_hotpot_to_squad_format(json.load(f))['data'] 
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
 
        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in self.file_path:  # only for the evaluation set 
            self.val_qid_string_to_int_map =                  {
                    entry["paragraphs"][0]['qas'][0]['id']: index
                    for index, entry in enumerate(self.data_json)
                }
        else:
            self.val_qid_string_to_int_map = None
            
            
    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        tensors_list = self.one_example_to_tensors(entry, idx)
        assert len(tensors_list) == 1
        return tensors_list[0]
    

    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        tensors_list = []
        for paragraph in example["paragraphs"]:  # example["paragraphs"] only contains one paragraph in hotpotqa
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c) # add a new token
                    else:
                        doc_tokens[-1] += c  # append the character to the last token
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                question_text = qa["question"] 
                sp_sent = qa["is_sup_fact"]
                sp_para = qa["is_supporting_para"]
                start_position = None
                end_position = None
                orig_answer_text = None
                
                p_list = list(locate(doc_tokens , lambda x: x == "<p>")) 
                assert(len(p_list) == len(sp_para))
                s_list = list(locate(doc_tokens , lambda x: x == "<s>"))
                assert(len(s_list) + len(p_list) == len(sp_sent) )
                
                # keep all answers in the document, not just the first matched answer. It also added the list of textual answers to make evaluation easy.
                answer_spans = []
                for answer in qa["answers"]:
                    orig_answer_text = answer["text"] 
                    answer_start = answer["answer_start"]
                    answer_end = answer["answer_end"]  
                    if(answer_start >= 0 and answer_end > 0):
                        try:
                            start_word_position = char_to_word_offset[answer_start]
                            end_word_position = char_to_word_offset[answer_end-1]
                        except:
                            print(f'error: Reading example {idx} failed')
                            start_word_position = -3
                            end_word_position = -3
                    else:
                        start_word_position = answer["answer_start"]
                        end_word_position = answer["answer_end"]
                    answer_spans.append({'start': start_word_position, 'end': end_word_position})

                    
                # ===== Given an example, convert it into tensors  =============
                query_tokens = self.tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:self.max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                
                # each original token in the context is tokenized to multiple sub_tokens
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    # hack: the line below should have been `self.tokenizer.tokenize(token')`
                    # but roberta tokenizer uses a different subword if the token is the beginning of the string
                    # or in the middle. So for all tokens other than the first, simulate that it is not the first
                    # token by prepending a period before tokenizing, then dropping the period afterwards
                    sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)
                
                # all sub tokens, truncate up to limit
                all_doc_tokens = all_doc_tokens[:self.max_doc_len-3]

                # The -3 accounts for [CLS], [q], [/q]  
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0
                if self.doc_stride < 0:                           # default
                    # negative doc_stride indicates no sliding window, but using first slice
                    self.doc_stride = -100 * len(all_doc_tokens)  # large -negtive value for the next loop to execute once
                
                # inputs to the model
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                start_positions_list = []
                end_positions_list = []
                q_type_list = []
                sp_sent_list =  [1 if ss else 0 for ss in sp_sent]
                sp_para_list = [1 if sp else 0 for sp in sp_para]
                
                for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):    # execute once by default
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = ["<cls>"] + ["<q>"] + query_tokens + ["</q>"] + doc_slice_tokens   
                    segment_ids = [0] * (len(query_tokens) + 3) + [1] *  len(doc_slice_tokens) 
                    assert len(segment_ids) == len(tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)   
                    input_mask = [1] * len(input_ids)

                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)

                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len

                    # ===== answer positions tensors  ============
                    doc_offset = len(query_tokens) + 3 - slice_start  # where context starts
                    start_positions = []
                    end_positions = []
                    q_type = None
                    assert(len(answer_spans) > 0)
                    for answer_span in answer_spans:
                        start_position = answer_span['start']   # reletive to context
                        end_position = answer_span['end']
                        if(start_position >= 0):
                            tok_start_position_in_doc = orig_to_tok_index[start_position]  # sub_tokens postion reletive to context
                            not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                            tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                            if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                                assert("this answer is outside the current slice")   # only has one slice with the large negative doc_stride
                                continue                                
                            start_positions.append(tok_start_position_in_doc + doc_offset)   # sub_tokens postion reletive to begining of all the tokens, including query sub tokens  
                            end_positions.append(tok_end_position_in_doc + doc_offset)
                            if(q_type != None and q_type != 0):
                                assert("inconsistance q_type")
                            q_type = 0
                
                        elif(start_position == -1):
                            if(q_type != None and q_type != 1):
                                assert("inconsistance q_type")
                            q_type = 1
                            start_positions.append(-1)  # -1 is the IGNORE_INDEX, will be ignored
                            end_positions.append(-1)     
                        elif(start_position == -2):
                            if(q_type != None and q_type != 2):
                                assert("inconsistance q_type")
                            q_type = 2
                            start_positions.append(-1)
                            end_positions.append(-1)     
                        else:
                            assert("unknown start_positions")
                            continue
                    assert len(start_positions) == len(end_positions)
                    
                    
                    if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                        continue

                    # answers from start_positions and end_positions if > self.max_num_answers
                    start_positions = start_positions[:self.max_num_answers]
                    end_positions = end_positions[:self.max_num_answers]

                    # -1 padding up to self.max_num_answers
                    padding_len = self.max_num_answers - len(start_positions)
                    start_positions.extend([-1] * padding_len)
                    end_positions.extend([-1] * padding_len)

                    # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                    found_start_positions = set()
                    found_end_positions = set()
                    for i, (start_position, end_position) in enumerate(zip(start_positions, end_positions)):
                        if start_position in found_start_positions:
                            start_positions[i] = -1
                        if end_position in found_end_positions:
                            end_positions[i] = -1
                        found_start_positions.add(start_position)
                        found_end_positions.add(end_position)

                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                    q_type_list.append(q_type)
                
                tensors_list.append((torch.tensor(input_ids_list), torch.tensor(input_mask_list), torch.tensor(segment_ids_list),
                                     torch.tensor(start_positions_list), torch.tensor(end_positions_list), torch.tensor(q_type_list),
                                      torch.tensor([sp_sent_list]),  torch.tensor([sp_para_list]),
                                     qa['id']))    
#                 tensors_list.append((doc_tokens))
        return tensors_list
 
    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 1  # qids  
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


# #### class hotpotqa

# In[109]:


class hotpotqa(pl.LightningModule):
    def __init__(self, args):
        super(hotpotqa, self).__init__()
        self.args = args
        self.hparams = args
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        num_new_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>", "<p>", "<q>", "</q>"]})
        self.tokenizer.model_max_length = self.args.max_seq_len
        self.model = self.load_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.num_labels = 2
        self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        
        self.dense_type = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.linear_type = torch.nn.Linear(self.model.config.hidden_size, 3)   #  question type (yes/no/span) classification  
        self.dense_sp_sent = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)	
        self.linear_sp_sent = torch.nn.Linear(self.model.config.hidden_size, 1)    	
        self.dense_sp_para = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.linear_sp_para = torch.nn.Linear(self.model.config.hidden_size, 1) 
        
        self.train_dataloader_object = self.val_dataloader_object  = None  # = self.test_dataloader_object = None
    
    def load_model(self):
        model = Longformer.from_pretrained(self.args.model_path)
        # config = LongformerConfig.from_pretrained('longformer-large-4096/') 
        # config.attention_mode = 'sliding_chunks'
        # model = Longformer.from_pretrained('longformer-base-4096')
        
        for layer in model.encoder.layer:
            layer.attention.self.attention_mode = self.args.attention_mode
            self.args.attention_window = layer.attention.self.attention_window

        print("Loaded model with config:")
        print(model.config)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        return model

#%%add_to hotpotqa    # does not seems to work for the @pl.data_loader decorator, missing which causes error "validation_step() takes 3 positional arguments but 4 were given"    
###################################################### dataloaders ########################################################### 
    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataloader_object is not None:
            return self.train_dataloader_object
        dataset = hotpotqaDataset(file_path=self.args.train_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=self.args.ignore_seq_with_no_answers)
 
        dl = DataLoader(dataset, batch_size=1, shuffle=False,   # set shuffle=False, otherwise it will sample a different subset of data every epoch with train_percent_check
                        num_workers=self.args.num_workers,  
                        collate_fn=hotpotqaDataset.collate_one_doc_and_lists)

        self.train_dataloader_object = dl
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        if self.val_dataloader_object is not None:
            return self.val_dataloader_object
        dataset = hotpotqaDataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=False)  # evaluation data should keep all examples 
        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, 
                        collate_fn=hotpotqaDataset.collate_one_doc_and_lists)
        self.val_dataloader_object = dl
        return self.val_dataloader_object

    # @pl.data_loader
    # def test_dataloader(self):
    #     if self.test_dataloader_object is not None:
    #         return self.test_dataloader_object
    #     dataset = hotpotqaDataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
    #                               max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
    #                               doc_stride=self.args.doc_stride,
    #                               max_num_answers=self.args.max_num_answers,
    #                               max_question_len=self.args.max_question_len,
    #                               ignore_seq_with_no_answers=False)  # evaluation data should keep all examples

    #     dl = DataLoader(dataset, batch_size=1, shuffle=False,
    #                     num_workers=self.args.num_workers, sampler=None,
    #                     collate_fn=hotpotqaDataset.collate_one_doc_and_lists)
    #     self.test_dataloader_object = dl
    #     return self.test_dataloader_object
 
    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, q_type, sp_sent, sp_para):
        if(input_ids.size(0) > 1):
            assert("multi rows per document")
        # Each batch is one document, and each row of the batch is a chunck of the document.  

        # local attention everywhere
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        
        # global attention for the cls and all question tokens
        question_end_index = self._get_special_index(input_ids, ["</q>"]) 
        attention_mask[:,:question_end_index[0].item()] = 2  # from <cls> until </q>
        
        # global attention for the sentence and paragraph special tokens  
        sent_indexes = self._get_special_index(input_ids, ["<p>", "<s>"])
        attention_mask[:, sent_indexes] = 2

        # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

        sequence_output = self.model(
                input_ids,
                attention_mask=attention_mask)[0]

        # The pretrained hotpotqa model wasn't trained with padding, so remove padding tokens
        # before computing loss and decoding.
        padding_len = input_ids[0].eq(self.tokenizer.pad_token_id).sum()
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len] 
        
        ###################################### layers on top of sequence_output ##################################
        

        ### 1. answer start and end positions classification ###   
        logits = self.qa_outputs(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1) 
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1)
 
        ### 2. type classification, similar as class LongformerClassificationHead(nn.Module) https://huggingface.co/transformers/_modules/transformers/modeling_longformer.html#LongformerForSequenceClassification.forward ### 
        type_logits = self.dense_type(sequence_output[:,0])	
        # Non-linearity	
        type_logits = torch.tanh(type_logits)  	
        type_logits = self.linear_type(type_logits)
        
        
        ### 3. supporting paragraph classification ### 
        p_index = self._get_special_index(input_ids, ["<p>"])
        sp_para_output = sequence_output[:,p_index,:]   
        sp_para_output_t = self.dense_sp_para(sp_para_output)  	
        # Non-linearity	
        sp_para_output_t = torch.tanh(sp_para_output_t)  	
        sp_para_output_t = self.linear_sp_para(sp_para_output_t) 

        # linear_sp_sent generates a single score for each sentence, instead of 2 scores for yes and no. 	
        # Argument the score with additional score=0. The same way did in the HOTPOTqa paper
        sp_para_output_aux = torch.zeros(sp_para_output_t.shape, dtype=torch.float, device=sp_para_output_t.device) 
        predict_support_para = torch.cat([sp_para_output_aux, sp_para_output_t], dim=-1).contiguous() 
 
        ### 4. supporting fact classification ###     
        sp_sent_output = sequence_output[:,sent_indexes,:]
        sp_sent_output_t = self.dense_sp_sent(sp_sent_output)   	
        # Non-linearity	
        sp_sent_output_t = torch.tanh(sp_sent_output_t)    	
        sp_sent_output_t = self.linear_sp_sent(sp_sent_output_t) 
        sp_sent_output_aux = torch.zeros(sp_sent_output_t.shape, dtype=torch.float, device=sp_sent_output_t.device) 
        predict_support_sent = torch.cat([sp_sent_output_aux, sp_sent_output_t], dim=-1).contiguous() 
        
        outputs = (start_logits, end_logits, type_logits, sp_para_output_t, sp_sent_output_t)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss  = self.loss_computation(start_positions, end_positions, start_logits, end_logits, q_type, type_logits, sp_para, predict_support_para, sp_sent, predict_support_sent)

        outputs = (answer_loss, type_loss, sp_para_loss, sp_sent_loss,) + outputs    
        return outputs
    
    def loss_computation(self, start_positions, end_positions, start_logits, end_logits, q_type, type_logits, sp_para, predict_support_para, sp_sent, predict_support_sent):
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            if not self.args.regular_softmax_loss:
                # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
                # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
                # but batch size is always 1, so this is not a problem
                start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1) 
                end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)

            else: 
                start_positions = start_positions[:, 0:1]   # only use the top1 start_position considering only one appearance of the answer string
                end_positions = end_positions[:, 0:1]
                start_loss = crossentropy(start_logits, start_positions[:, 0])
                end_loss = crossentropy(end_logits, end_positions[:, 0])
                 
            crossentropy = torch.nn.CrossEntropyLoss(ignore_index=-1)
            type_loss = crossentropy(type_logits, q_type)  
            
            crossentropy_average = torch.nn.CrossEntropyLoss(reduction = 'mean', ignore_index=-1)      
            sp_para_loss = crossentropy_average(predict_support_para.view(-1, 2), sp_para.view(-1))
            sp_sent_loss = crossentropy_average(predict_support_sent.view(-1, 2), sp_sent.view(-1))      
                
            answer_loss = (start_loss + end_loss) / 2 
        return answer_loss, type_loss, sp_para_loss, sp_sent_loss  

 
    def _get_special_index(self, input_ids, special_tokens):
        assert(input_ids.size(0)==1) 
        mask = input_ids != input_ids # initilaize 
        for special_token in special_tokens:
            mask = torch.logical_or(mask, input_ids.eq(self.tokenizer.convert_tokens_to_ids(special_token)))  
        token_indices = torch.nonzero(mask)    
        
        return token_indices[:,1]    

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0) 
        
        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target, considing only one correct target 
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer, considing more correct targets

        # target are indexes of tokens, padded with ignore_index=-1
        # logits are scores (one for each label) for each token

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the masked targets
        masked_target = target * (1 - target_mask.long())                   
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)   
        
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')                      # padding logits are all replaced by -inf    tensor([[0.4382,   -inf,   -inf,   -inf,   -inf,...]])
        
        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)

        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False) 
        
        # compute the loss
        loss = -(log_score - log_norm) 
        
        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`: when computing start_loss and end_loss for question with the gold answer of yes/no 
        # when `target` is all `ignore_index`, loss is 0 
        loss = loss[~torch.isinf(loss)].sum() 
        return loss 
 

    # A hook to overwrite to define your own DDP(DistributedDataParallel) implementation init. 
    # The only requirement is that: 
    # 1. On a validation batch the call goes to model.validation_step.
    # 2. On a training batch the call goes to model.training_step.
    # 3. On a testing batch, the call goes to model.test_step
    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def configure_optimizers(self):
        # Set up optimizers and (optionally) learning rate schedulers
        def lr_lambda(current_step):
            if current_step < self.args.warmup:
                return float(current_step) / float(max(1, self.args.warmup))
            return max(0.0, float(self.args.steps - current_step) / float(max(1, self.args.steps - self.args.warmup)))

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)  # scheduler is not saved in the checkpoint, but global_step is, which is enough to restart
        self.scheduler.step(self.global_step)

        return optimizer
    

    # A hook to do a lot of non-standard training tricks such as learning-rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.scheduler.step(self.global_step)

    def on_epoch_start(self):
        print("Start epoch ", self.current_epoch)
        pass
    def training_step(self, batch, batch_nb):
        # do the forward pass and calculate the loss for a batch  
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para, qids = batch 
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss  = output[:4]
        loss = answer_loss + type_loss + sp_para_loss + sp_sent_loss
        
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']  # loss.new_zeros(1) is tensor([0.]), converting 'lr' to tensor' by adding it. 
        tensorboard_logs = {'train_answer_loss': answer_loss, 'train_type_loss': type_loss, 'train_sp_para_loss': sp_para_loss, 'train_sp_sent_loss': sp_sent_loss, 'lr': lr,  
                            'input_size': torch.tensor(input_ids.numel()).type_as(loss) ,
                            'mem': torch.tensor(torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3).type_as(loss) }
                            
        return {'loss': loss, 'log': tensorboard_logs}

    # When the validation_step is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.
    def validation_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para, qids = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output = output 
        loss = answer_loss +  type_loss + sp_para_loss + sp_sent_loss
        answers_pred, sp_sent_pred, sp_para_pred = self.decode(input_ids, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output)

        if(len(answers_pred) != 1):
            print("len(answers_pred) != 1")
            assert(len(answers_pred) == 1)
        answers_pred = answers_pred[0]
        answer_score = answers_pred['score']  # (start_logit + end_logit + p_type_score) / 3

        if(q_type == 1):
            answer_gold = 'yes'
        elif(q_type == 2):
            answer_gold = 'no' 
        else:
            # even though there can be multiple gold start_postion (subword_start) and end_position(subword_end), the corresponing answer string are same
            answer_gold_token_ids = input_ids[0, subword_starts[0][0]: subword_ends[0][0] + 1] 
            answer_gold_tokens = self.tokenizer.convert_ids_to_tokens(answer_gold_token_ids.tolist()) 
            answer_gold = self.tokenizer.convert_tokens_to_string(answer_gold_tokens) 

        f1, prec, recall = self.f1_score(answers_pred['text'], answer_gold)
        em = self.exact_match_score(answers_pred['text'], answer_gold) 
        f1 = torch.tensor(f1).type_as(loss)
        prec = torch.tensor(prec).type_as(loss)
        recall = torch.tensor(recall).type_as(loss)
        em = torch.tensor(em).type_as(loss)

        if(len(sp_sent_pred) > 0):
            sp_sent_em, sp_sent_precision, sp_sent_recall, sp_sent_f1 = self.sp_metrics(sp_sent_pred, torch.where(sp_sent.squeeze())[0].tolist())
            sp_sent_em = torch.tensor(sp_sent_em).type_as(loss)
            sp_sent_precision = torch.tensor(sp_sent_precision).type_as(loss)
            sp_sent_recall = torch.tensor(sp_sent_recall).type_as(loss)
            sp_sent_f1 = torch.tensor(sp_sent_f1).type_as(loss)

            joint_prec = prec * sp_sent_precision
            joint_recall = recall * sp_sent_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = torch.tensor(0.0).type_as(loss)
            joint_em = em * sp_sent_em 

        else:
            sp_sent_em, sp_sent_precision, sp_sent_recall, sp_sent_f1 = torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss)
            joint_em, joint_f1, joint_prec, joint_recall =  torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss)


        return { 'vloss': loss, 'answer_loss': answer_loss, 'type_loss': type_loss, 'sp_para_loss': sp_para_loss, 'sp_sent_loss': sp_sent_loss,
                'answer_score': answer_score, 'f1': f1, 'prec':prec, 'recall':recall, 'em': em,
                'sp_em': sp_sent_em, 'sp_f1': sp_sent_f1, 'sp_prec': sp_sent_precision, 'sp_recall': sp_sent_recall,
                'joint_em': joint_em, 'joint_f1': joint_f1, 'joint_prec': joint_prec, 'joint_recall': joint_recall}



    def decode(self, input_ids, start_logits, end_logits, type_logits, sp_para_logits, sp_sent_logits):

        question_end_index = self._get_special_index(input_ids, ["</q>"]) 
        
        # one example per batch
        start_logits = start_logits.squeeze()
        end_logits = end_logits.squeeze() 
        start_logits_indices = start_logits.topk(k=self.args.n_best_size, dim=-1).indices 
        end_logits_indices = end_logits.topk(k=self.args.n_best_size, dim=-1).indices 
        if(len(start_logits_indices.size()) > 1):
            print("len(start_logits_indices.size()): ", len(start_logits_indices.size()))
            assert("len(start_logits_indices.size()) > 1")
        p_type = torch.argmax(type_logits, dim=1).item()
        p_type_score = torch.max(type_logits, dim=1)[0]  
        
        answers = []
        if p_type == 0:
            potential_answers = []
            for start_logit_index in start_logits_indices: 
                for end_logit_index in end_logits_indices: 
                    if start_logit_index <= question_end_index.item():
                        continue
                    if end_logit_index <= question_end_index.item():
                        continue
                    if start_logit_index > end_logit_index:
                        continue
                    answer_len = end_logit_index - start_logit_index + 1
                    if answer_len > self.args.max_answer_length:
                        continue
                    potential_answers.append({'start': start_logit_index, 'end': end_logit_index,
                                              'start_logit': start_logits[start_logit_index],  # single logit score for start position at start_logit_index
                                              'end_logit': end_logits[end_logit_index]})    
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)  
            if len(sorted_answers) == 0:
                answers.append({'text': 'NoAnswerFound', 'score': -1000000})
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[0, answer['start']: answer['end'] + 1]
                answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                text = self.tokenizer.convert_tokens_to_string(answer_tokens) 
                score = (torch.sigmoid(answer['start_logit']) + torch.sigmoid(answer['end_logit']) + torch.sigmoid(p_type_score)) / 3
                answers.append({'text': text, 'score': score})
                # print("answers: " + str(answers))
        elif p_type == 1: 
            answers.append({'text': 'yes', 'score': p_type_score})
        elif p_type == 2:
            answers.append({'text': 'no', 'score': p_type_score})
        else:
            assert False 
    
        p_index = self._get_special_index(input_ids, ["<p>"]) 
        sent_indexes = self._get_special_index(input_ids, ["<p>", "<s>"])
        
        s_to_p_map = []
        for s in sent_indexes:
            s_to_p = torch.where(torch.le(p_index, s))[0][-1]     # last p_index smaller or equal to s
            s_to_p_map.append(s_to_p.item())  
        sp_para_top2 = sp_para_logits.squeeze().topk(k=2).indices
        if(sp_sent_logits.squeeze().size(0) > 12):
            sp_sent_top12 = sp_sent_logits.squeeze().topk(k=12).indices
        else:
            sp_sent_top12 = sp_sent_logits.squeeze().topk(k=sp_sent_logits.squeeze().size(0)).indices 
        
        sp_sent_pred = set()
        sp_para_pred = set()
        for sp_sent in sp_sent_top12:
            sp_sent_to_para = s_to_p_map[sp_sent.item()]
            if sp_sent_to_para in sp_para_top2:
                sp_sent_pred.add(sp_sent.item())
                sp_para_pred.add(sp_sent_to_para)  
        return (answers, sp_sent_pred, sp_para_pred)

    def normalize_answer(self, s):

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)
        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall


    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))


    def sp_metrics(self, prediction, gold): 
        tp, fp, fn = 0, 0, 0
        for e in prediction:
            if e in gold:
                tp += 1
            else:
                fp += 1
        for e in gold:
            if e not in prediction:
                fn += 1

        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0

        return em, prec, recall, f1
 

    # If a validation_step is not defined, this won't be called. Called at the end of the validation loop with the outputs of validation_step.
    def validation_end(self, outputs):
        print("validation_epoch_end")
        avg_loss = torch.stack([x['vloss'] for x in outputs]).mean() 
        avg_answer_loss = torch.stack([x['answer_loss'] for x in outputs]).mean() 
        avg_type_loss = torch.stack([x['type_loss'] for x in outputs]).mean() 
        avg_sp_para_loss = torch.stack([x['sp_para_loss'] for x in outputs]).mean() 
        avg_sp_sent_loss = torch.stack([x['sp_sent_loss'] for x in outputs]).mean() 
        
        answer_scores = [x['answer_score'] for x in outputs]  
        f1_scores = [x['f1'] for x in outputs]  
        em_scores = [x['em'] for x in outputs]  
        prec_scores =  [x['prec'] for x in outputs] 
        recall_scores = [x['recall'] for x in outputs]  
        sp_sent_f1_scores = [x['sp_f1'] for x in outputs]  
        sp_sent_em_scores = [x['sp_em'] for x in outputs] 
        sp_sent_prec_scores = [x['sp_prec'] for x in outputs]  
        sp_sent_recall_scores = [x['sp_recall'] for x in outputs] 
        joint_f1_scores = [x['joint_f1'] for x in outputs]    
        joint_em_scores = [x['joint_em'] for x in outputs]    
        joint_prec_scores = [x['joint_prec'] for x in outputs]   
        joint_recall_scores = [x['joint_recall'] for x in outputs]   
        print(f'before sync --> sizes:  {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')
        
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size 
            torch.distributed.all_reduce(avg_answer_loss, op=torch.distributed.ReduceOp.SUM)
            avg_answer_loss /= self.trainer.world_size 
            torch.distributed.all_reduce(avg_type_loss, op=torch.distributed.ReduceOp.SUM)
            avg_type_loss /= self.trainer.world_size 
            torch.distributed.all_reduce(avg_sp_para_loss, op=torch.distributed.ReduceOp.SUM)
            avg_sp_para_loss /= self.trainer.world_size 
            torch.distributed.all_reduce(avg_sp_sent_loss, op=torch.distributed.ReduceOp.SUM)
            avg_sp_sent_loss /= self.trainer.world_size 
              
            # int_qids = self.sync_list_across_gpus(int_qids, avg_loss.device, torch.int)
            answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.float)
            prec_scores = self.sync_list_across_gpus(prec_scores, avg_loss.device, torch.float)
            recall_scores = self.sync_list_across_gpus(recall_scores, avg_loss.device, torch.float)

            sp_sent_f1_scores = self.sync_list_across_gpus(sp_sent_f1_scores, avg_loss.device, torch.float)
            sp_sent_em_scores = self.sync_list_across_gpus(sp_sent_em_scores, avg_loss.device, torch.float)
            sp_sent_prec_scores = self.sync_list_across_gpus(sp_sent_prec_scores, avg_loss.device, torch.float)
            sp_sent_recall_scores = self.sync_list_across_gpus(sp_sent_recall_scores, avg_loss.device, torch.float)

            joint_f1_scores = self.sync_list_across_gpus(joint_f1_scores, avg_loss.device, torch.float)
            joint_em_scores = self.sync_list_across_gpus(joint_em_scores, avg_loss.device, torch.float)
            joint_prec_scores = self.sync_list_across_gpus(joint_prec_scores, avg_loss.device, torch.float)
            joint_recall_scores = self.sync_list_across_gpus(joint_recall_scores, avg_loss.device, torch.float)
        print(f'after sync --> sizes: {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')

        avg_val_f1 = sum(f1_scores) / len(f1_scores) 
        avg_val_em = sum(em_scores) / len(em_scores) 
        avg_val_prec = sum(prec_scores) / len(prec_scores) 
        avg_val_recall = sum(recall_scores) / len(recall_scores)  

        avg_val_sp_sent_f1 = sum(sp_sent_f1_scores) / len(sp_sent_f1_scores) 
        avg_val_sp_sent_em = sum(sp_sent_em_scores) / len(sp_sent_em_scores) 
        avg_val_sp_sent_prec = sum(sp_sent_prec_scores) / len(sp_sent_prec_scores) 
        avg_val_sp_sent_recall = sum(sp_sent_recall_scores) / len(sp_sent_recall_scores)  

        avg_val_joint_f1 = sum(joint_f1_scores) / len(joint_f1_scores) 
        avg_val_joint_em = sum(joint_em_scores) / len(joint_em_scores) 
        avg_val_joint_prec = sum(joint_prec_scores) / len(joint_prec_scores) 
        avg_val_joint_recall = sum(joint_recall_scores) / len(joint_recall_scores)  
 
        print("avg_loss: ", avg_loss, end = '\t') 
        print("avg_answer_loss: ", avg_answer_loss, end = '\t') 
        print("avg_type_loss: ", avg_type_loss, end = '\t') 
        print("avg_sp_para_loss: ", avg_sp_para_loss, end = '\t') 
        print("avg_sp_sent_loss: ", avg_sp_sent_loss)  
        
        print("avg_val_f1: ", avg_val_f1, end = '\t')
        print("avg_val_em: ", avg_val_em, end = '\t')
        print("avg_val_prec: ", avg_val_prec, end = '\t')
        print("avg_val_recall: ", avg_val_recall)
        
        print("avg_val_sp_sent_f1: ", avg_val_sp_sent_f1, end = '\t')
        print("avg_val_sp_sent_em: " , avg_val_sp_sent_em, end = '\t')
        print("avg_val_sp_sent_prec: ", avg_val_sp_sent_prec, end = '\t')
        print("avg_val_sp_sent_recall: ", avg_val_sp_sent_recall)
        
        print("avg_val_joint_f1: " , avg_val_joint_f1, end = '\t')
        print("avg_val_joint_em: ", avg_val_joint_em, end = '\t')
        print("avg_val_joint_prec: ", avg_val_joint_prec, end = '\t')
        print("avg_val_joint_recall: ", avg_val_joint_recall) 

        logs = {'avg_val_loss': avg_loss, 'avg_val_answer_loss': avg_answer_loss, 'avg_val_type_loss': avg_type_loss, 'avg_val_sp_para_loss': avg_sp_para_loss, 'avg_val_sp_sent_loss': avg_sp_sent_loss, 
                'avg_val_f1': avg_val_f1, 'avg_val_em': avg_val_em,  'avg_val_prec': avg_val_prec, 'avg_val_recall': avg_val_recall,
                'avg_val_sp_sent_f1': avg_val_sp_sent_f1, 'avg_val_sp_sent_em': avg_val_sp_sent_em,  'avg_val_sp_sent_prec': avg_val_sp_sent_prec, 'avg_val_sp_sent_recall': avg_val_sp_sent_recall,
                'avg_val_joint_f1': avg_val_joint_f1, 'avg_val_joint_em': avg_val_joint_em,  'avg_val_joint_prec': avg_val_joint_prec, 'avg_val_joint_recall': avg_val_joint_recall
               }

        return {'avg_val_loss': avg_loss, 'log': logs}

    def sync_list_across_gpus(self, l, device, dtype):
        l_tensor = torch.tensor(l, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='hotpotqa-longformer')
        parser.add_argument("--save_prefix", type=str, required=True)
        parser.add_argument("--train_dataset", type=str, required=False, help="Path to the training squad-format")
        parser.add_argument("--dev_dataset", type=str, required=True, help="Path to the dev squad-format")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--gpus", type=str, default='0',
                            help="Comma separated list of gpus. Default is gpu 0. To use CPU, use --gpus "" ")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00005, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1, help="How often within one training epoch to check the validation set.")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
        parser.add_argument("--max_seq_len", type=int, default=4096,
                            help="Maximum length of seq passed to the transformer model")
        parser.add_argument("--max_doc_len", type=int, default=4096,
                            help="Maximum number of wordpieces of the input document")
        parser.add_argument("--max_num_answers", type=int, default=64,
                            help="Maximum number of answer spans per document (64 => 94%)")
        parser.add_argument("--max_question_len", type=int, default=55,
                            help="Maximum length of the question")
        parser.add_argument("--doc_stride", type=int, default=-1,
                            help="Overlap between document chunks. Use -1 to only use the first chunk")
        parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                            help="each example should have at least one answer. Default is False")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--n_best_size", type=int, default=20,
                            help="Number of answer candidates. Used at decoding time")
        parser.add_argument("--max_answer_length", type=int, default=30,
                            help="maximum num of wordpieces/answer. Used at decoding time")
        parser.add_argument("--regular_softmax_loss", action='store_true', help="IF true, use regular softmax. Default is using ORed softmax loss")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_path", type=str,
                            help="Path to the checkpoint directory")
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--attention_mode", type=str, choices=['tvm', 'sliding_chunks'],
                            default='sliding_chunks', help='Which implementation of selfattention to use')
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument('--train_percent', type=float, default=1.0)
        return parser


# ### main

# In[111]:


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) 

    hotpotqa.__abstractmethods__=set()   # without this, got an error "Can't instantiate abstract class hotpotqa with abstract methods" if these two abstract methods are not implemented in the same cell where class hotpotqa defined 
    model = hotpotqa(args)
    model.to('cuda')    # this is necessary to use gpu

    logger = TestTubeLogger( # The TestTubeLogger adds a nicer folder structure to manage experiments and snapshots all hyperparameters you pass to a LightningModule.
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_f1',
        mode='max',
        prefix=''
    )

    args.gpus = [int(x) for x in args.gpus.split(',')] if args.gpus is not "" else None
    num_devices = len(args.gpus) #1 or len(args.gpus)
    train_set_size = 90447 * args.train_percent    # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    args.steps = args.epochs * train_set_size / (args.batch_size * num_devices)

     
    print(f'>>>>>>> #train_set_size: {train_set_size}, #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * num_devices} <<<<<<<')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if args.gpus and (len(args.gpus) > 1) else None,
                         track_grad_norm=-1, max_epochs=args.epochs, early_stop_callback=None,
                         accumulate_grad_batches=args.batch_size,
                         train_percent_check = args.train_percent,
                         val_percent_check=args.val_percent_check,
                        #  test_percent_check=args.val_percent_check,
                         logger=logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O1',
                         check_val_every_n_epoch=1
                         )
    
    trainer.fit(model)


import argparse
if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="hotpotqa")
    parser = hotpotqa.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    for arg in vars(args):
        print((arg, getattr(args, arg)))
    main(args)


# In[ ]:


# --train_dataset hotpot_train_v1.1.json --dev_dataset hotpot_dev_distractor_v1.json  \
#     --gpus 0  --num_workers 4 \
#     --max_seq_len 4096 --doc_stride -1  \
#     --train_percent 0.003 \
#     --save_prefix hotpotqa-longformer  --model_path longformer-base-4096 --test

