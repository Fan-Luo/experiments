
# coding: utf-8

# ### convert hotpotqa to squard format

# According to Longformer: use the following input format with special tokens:  “[CLS] [q] question [/q] [p] sent1,1 [s] sent1,2 [s] ... [p] sent2,1 [s] sent2,2 [s] ...” 
# where [s] and [p] are special tokens representing sentences and paragraphs. The special tokens were added to the RoBERTa vocabulary and randomly initialized before task finetuning.

# In[ ]:


# helper functions to convert hotpotqa to squard format modified from  https://github.com/chiayewken/bert-qa/blob/master/run_hotpot.py
import ctypes   
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import tqdm 
from datetime import datetime 
import pytz 
timeZ_Az = pytz.timezone('US/Mountain') 

QUESTION_START = '[question]'
QUESTION_END = '[/question]' 
TITLE_START = '<t>'  # indicating the start of the title of a paragraph (also used for loss over paragraphs)
TITLE_END = '</t>'   # indicating the end of the title of a paragraph
SENT_MARKER_END = '[/sent]'  # indicating the end of the title of a sentence (used for loss over sentences)
PAR = '[/par]'  # used for indicating end of the regular context and beginning of `yes/no/null` answers 

def create_example_dict(context, answer, id, question, is_sup_fact, is_supporting_para):
    return {
        "context": context,
        "qas": [                        # each context corresponds to only one qa in hotpotqa
            {
                "answer": answer,
                "id": id,
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


# In[ ]:


import re
import string

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
            
        is_supporting_para = []  # a boolean list with 10 True/False elements, one for each paragraph
        is_sup_fact = []         # a boolean list with True/False elements, one for each context sentence
        for para_title, para_lines in raw_contexts:
            is_supporting_para.append(para_title in support_para)   
            for sent_id, sent in enumerate(para_lines):
                is_sup_fact.append( (para_title, sent_id) in sp_set )    
        
        contexts = []   
        for para_id, para in enumerate(raw_contexts):   
            title = _normalize_text(para[0])    
            sents = [_normalize_text(sent) for sent in para[1]] 
            
            if("kept_para_sent" in example):    # reduceded context 
                sent_joint = '' 
                for sent_id, sent in enumerate(sents):  
                    if(sent_id > 0 and example["kept_para_sent"][para_id][sent_id] - example["kept_para_sent"][para_id][sent_id-1] > 1):    
                        sent_joint += (' </s> ' + sent + ' ' + SENT_MARKER_END )   # </s> indicates at least one sentence is removed    
                    else:   
                        sent_joint += (sent + ' ' + SENT_MARKER_END )   
            else:   
                sent_joint =  (' ' + SENT_MARKER_END +' ').join(sents) + ' ' + SENT_MARKER_END      
                
            contexts.append(TITLE_START + ' ' + title + ' ' + TITLE_END + ' ' +  sent_joint)    
            
        # contexts = [TITLE_START + ' ' + lst[0]  + ' ' + TITLE_END + ' ' + (' ' + SENT_MARKER_END +' ').join(lst[1]) + ' ' + SENT_MARKER_END for lst in raw_contexts]    
 
        answer = _normalize_text(example["answer"])  
        
        if(len(answer) > 0):   # answer can be '' after normalize
            new_dict["data"].append(
                create_para_dict(
                    create_example_dict(
                        context=contexts,
                        answer=answer,
                        id = example["_id"],
                        question=_normalize_text(example["question"]),
                        is_sup_fact = is_sup_fact,
                        is_supporting_para = is_supporting_para 
                    )
                )
            ) 
#     print("number of questions with answer not found in context: ", num_null_answer)
#     print("number of questions with answer 'yes': ", num_yes_answer)
#     print("number of questions with answer 'no': ", num_no_answer)
    return new_dict

def _normalize_text(s):  # copied from the official triviaqa repo
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


# ### longfomer's fine-tuning
# 
# 
# - For answer span extraction we use BERT’s QA model with addition of a question type (yes/no/span) classification head over the first special token ([CLS]).
# 
# - For evidence extraction we apply 2 layer feedforward networks on top of the representations corresponding to sentence and paragraph tokens to get the corresponding evidence prediction scores and use binary cross entropy loss to train the model.
# 
# - We combine span, question classification, sentence, and paragraphs losses and train the model in a multitask way using linear combination of losses.
# 

# In[ ]:


### Section2: This is modified from longfomer's fine-tuning with triviaqa.py from https://github.com/allenai/longformer/blob/master/scripts/triviaqa.py
# !conda install transformers --yes
# !conda install cudatoolkit=10.0 --yes
# !python -m pip install git+https://github.com/allenai/longformer.git
####requirements.txt:torch>=1.2.0, transformers>=3.0.2, tensorboardX, pytorch-lightning==0.6.0, test-tube==0.7.5
# !conda install -c conda-forge regex --force-reinstall --yes
# !conda install pytorch-lightning -c conda-forge
# !pip install jdc 
# !pip install test-tube 
# !conda install ipywidgets --yes
# !conda update --force conda --yes  
# !jupyter nbextension enable --py widgetsnbextension 
# !conda install jupyter --yes

# need to run this every time start this notebook, to add python3.7/site-packages to sys.pat, in order to import ipywidgets, which is used when RobertaTokenizer.from_pretrained('roberta-base') 
# import sys
# sys.path.insert(-1, '/xdisk/msurdeanu/fanluo/miniconda3/lib/python3.7/site-packages')

import os
from collections import defaultdict
import json
import string
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.logging import TestTubeLogger    # sometimes pytorch_lightning.loggers works instead


from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from more_itertools import locate
from collections import Counter


# In[ ]:


print(pl.__file__)


# #### class hotpotqaDataset

# ##### \_\_init\_\_, \_\_getitem\_\_ and \_\_len\_\_ 

# In[ ]:


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

#         print(tokenizer.all_special_tokens) 
    
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
        if(len(tensors_list) != 1):
            print("tensors_list: ", tensors_list)
        assert len(tensors_list) == 1
        return tensors_list[0]
    


# ##### one_example_to_tensors

# In[ ]:

    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        
        def map_answer_positions(char_to_word_offset, orig_to_tok_index, answer_start, answer_end, slice_start, slice_end, doc_offset):
            
            # char offset to word offset
            if(answer_start >= len(char_to_word_offset)):
                print("answer_start: ", answer_start)
                print("len(char_to_word_offset): ", len(char_to_word_offset))
            start_word_position = char_to_word_offset[answer_start]
            end_word_position = char_to_word_offset[answer_end-1] 

#             print("start_word_position: ", start_word_position)
#             print("end_word_position: ", end_word_position)
            # sub_tokens postion reletive to context
            tok_start_position_in_doc = orig_to_tok_index[start_word_position]  
            not_end_of_doc = int(end_word_position + 1 < len(orig_to_tok_index))
            tok_end_position_in_doc = orig_to_tok_index[end_word_position + not_end_of_doc] - not_end_of_doc
            
            if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                return (-1, -1) # this answer is outside the current slice                     
            
            # sub_tokens postion reletive to begining of all the tokens, including query sub tokens  
            start_position = tok_start_position_in_doc + doc_offset  
            end_position = tok_end_position_in_doc + doc_offset
            
            return (start_position, end_position)
        
        tensors_list = []
        for paragraph in example["paragraphs"]:  # example["paragraphs"] only contains one paragraph in hotpotqa
            context = self.tokenizer.sep_token + ' ' + (' ' + self.tokenizer.sep_token + ' ').join(paragraph["context"] )
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c) # add a new token
                    else:
                        doc_tokens[-1] += c  # append the character to the last token
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            
#             print("len(char_to_word_offset): ", len(char_to_word_offset))
#             print("char_to_word_offset: ", char_to_word_offset)
            for qa in paragraph["qas"]:
                question_text = qa["question"]
                answer = _normalize_text(qa["answer"])
#                 print("question text: ", question_text)  
                sp_sent = qa["is_sup_fact"]
                sp_para = qa["is_supporting_para"]
                start_position = None
                end_position = None 
                
                   
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
                all_doc_tokens = all_doc_tokens[:self.max_doc_len-7] 

                # The -7 accounts for CLS, QUESTION_START, QUESTION_END， [/par]， yes， no， </s>   
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 7
                if(max_tokens_per_doc_slice <= 0):
                    print("(max_tokens_per_doc_slice <= 0)")
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
                
                    # print("slice_start in range") 
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = [self.tokenizer.cls_token] + [QUESTION_START] + query_tokens + [QUESTION_END] + doc_slice_tokens + [PAR] + self.tokenizer.tokenize("yes") + self.tokenizer.tokenize("no") + [self.tokenizer.eos_token]   
                    segment_ids = [0] * (len(query_tokens) + 3) + [1] * (len(doc_slice_tokens) + 4) 
#                     if(len(segment_ids) != len(tokens)):
#                         print("len(segment_ids): ", len(segment_ids))
#                         print("len(tokens): ", len(tokens))
                    assert len(segment_ids) == len(tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)   
                    input_mask = [1] * len(input_ids)

                    doc_offset = len(query_tokens) + 3 - slice_start  # where context starts
                    
                    # ===== answer positions tensors  ============
                    start_positions = []
                    end_positions = []
 
                    answer = qa["answer"] 
                    # print("idx: ", idx, " qa['id']: ", qa['id'], " answer: ", answer)
                    if answer == '':
                        q_type = -1
                        start_positions.append(-1)   
                        end_positions.append(-1)           
                    
                    elif answer == 'yes':
                        q_type = 1
                        start_positions.append(len(tokens)-3)   
                        end_positions.append(len(tokens)-3) 
                    elif answer == 'no':
                        q_type = 2
                        start_positions.append(len(tokens)-2)   
                        end_positions.append(len(tokens)-2)  
                    else:
                        # keep all the occurences of answer in the context 
#                         for m in re.finditer("\s?".join(answer.split()), context):   # "\s?".join(answer.split()) in order to match even with extra space in answer or context
                        for m in re.finditer(_normalize_text(answer), context, re.IGNORECASE):
                            answer_start, answer_end = m.span() 
                            start_position, end_position = map_answer_positions(char_to_word_offset, orig_to_tok_index, answer_start, answer_end, slice_start, slice_end, doc_offset)
                            if(start_position != -1):
                                start_positions.append(start_position)   
                                end_positions.append(end_position)
                            
                        if(len(start_positions) > 0): 
                            q_type = 0
                        else: # answer not found in context
                            q_type = -1
                            start_positions.append(-1)   
                            end_positions.append(-1) 


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
                    
                                         
                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)
                        
                        print("self.doc_stride >= 0")
                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len  
                        
                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                    q_type_list.append(q_type)
                    
                tensors_list.append((torch.tensor(input_ids_list), torch.tensor(input_mask_list), torch.tensor(segment_ids_list),
                                     torch.tensor(start_positions_list), torch.tensor(end_positions_list), torch.tensor(q_type_list),
                                      torch.tensor([sp_sent_list]),  torch.tensor([sp_para_list]),
                                     qa['id'], answer))     
        return tensors_list


# ##### collate_one_doc_and_lists

# In[ ]:



    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 2  # qid and answer  
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


# #### class hotpotqa

# ##### \_\_init\_\_,  forward, dataloaders

# In[ ]:


class hotpotqa(pl.LightningModule):
    def __init__(self, args):
        super(hotpotqa, self).__init__()
        self.args = args
        self.hparams = args
 
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        num_new_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": [TITLE_START, TITLE_END, SENT_MARKER_END, QUESTION_START , QUESTION_END, PAR]})
#         print(self.tokenizer.all_special_tokens)
        self.tokenizer.model_max_length = self.args.max_seq_len
        self.model_1 = self.load_model()
        self.model_1.resize_token_embeddings(len(self.tokenizer))
        
        self.num_labels = 2
        self.qa_outputs_1 = torch.nn.Linear(self.model_1.config.hidden_size, self.num_labels)
        self.linear_type_1 = torch.nn.Linear(self.model_1.config.hidden_size, 3) 
        self.fnn_sp_sent_1 = torch.nn.Sequential(
          torch.nn.Linear(self.model_1.config.hidden_size, self.model_1.config.hidden_size), 
          torch.nn.GELU(),
          torch.nn.Linear(self.model_1.config.hidden_size, 1),      # score for 'yes', while 0 for 'no'
        )        
         
        self.fnn_sp_para_1 = torch.nn.Sequential(
          torch.nn.Linear(self.model_1.config.hidden_size, self.model_1.config.hidden_size), 
          torch.nn.GELU(),
          torch.nn.Linear(self.model_1.config.hidden_size, 1),      # score for 'yes', while 0 for 'no'
        ) 
         
        self.model_2 = self.load_model()
        self.model_2.resize_token_embeddings(len(self.tokenizer))
        
        self.qa_outputs_2 = torch.nn.Linear(self.model_2.config.hidden_size, self.num_labels)
        self.linear_type_2 = torch.nn.Linear(self.model_2.config.hidden_size, 3)   #  question type (yes/no/span) classification 
       
        self.fnn_sp_sent_2 = torch.nn.Sequential(
          torch.nn.Linear(self.model_2.config.hidden_size, self.model_2.config.hidden_size), 
          torch.nn.GELU(),
          torch.nn.Linear(self.model_2.config.hidden_size, 1),      # score for 'yes', while 0 for 'no'
        )
        
        self.fnn_sp_para_2 = torch.nn.Sequential(
          torch.nn.Linear(self.model_2.config.hidden_size, self.model_2.config.hidden_size), 
          torch.nn.GELU(),
          torch.nn.Linear(self.model_2.config.hidden_size, 1),      # score for 'yes', while 0 for 'no'
        )
        
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
    
    def load_model(self):
        
        # config = LongformerConfig.from_pretrained(self.args.model_path) 
        # config.attention_mode = self.args.attention_mode
        # model = Longformer.from_pretrained(self.args.model_path, config=config)
        
        if 'longformer' in self.args.model_path:
            model = Longformer.from_pretrained(self.args.model_path) 

            for layer in model.encoder.layer:
                layer.attention.self.attention_mode = self.args.attention_mode
                self.args.attention_window = layer.attention.self.attention_window
        else:
            model = AutoModel.from_pretrained(self.args.model_path)
            
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
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=(sampler is None),
                        num_workers=self.args.num_workers, sampler=sampler,
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
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=hotpotqaDataset.collate_one_doc_and_lists)
        self.val_dataloader_object = dl
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        if self.test_dataloader_object is not None:
            return self.test_dataloader_object
        dataset = hotpotqaDataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=False)  # evaluation data should keep all examples
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=dist_sampler,
                        collate_fn=hotpotqaDataset.collate_one_doc_and_lists)
        self.test_dataloader_object = dl
        return self.test_dataloader_object


#%%add_to hotpotqa  
    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, q_type, sp_sent, sp_para):

        if(input_ids.size(0) > 1):
            print("multi rows per document")
        assert(input_ids.size(0)==1)
        # Each batch is one document, and each row of the batch is a chunck of the document.    ????
        # Make sure all rows have the same question length.
        
        ########################################################## stage 1 ############################################################################
        # local attention everywhere
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        
        # global attention for the cls and all question tokens
        question_end_index = self._get_special_index(input_ids, [QUESTION_END])  
        attention_mask[:,:question_end_index[0].item()+1] = 2  # from <cls> until </q> 
        
        # global attention for the sentence and paragraph special tokens  
        sent_indexes = self._get_special_index(input_ids, [SENT_MARKER_END])
        attention_mask[:, sent_indexes] = 2
        
        para_indexes = self._get_special_index(input_ids, [TITLE_START])
        attention_mask[:, para_indexes] = 2       
         

        # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

        sequence_output = self.model_1(
                input_ids,
                attention_mask=attention_mask)[0]
#         print("size of sequence_output: " + str(sequence_output.size()))

        # The pretrained hotpotqa model wasn't trained with padding, so remove padding tokens
        # before computing loss and decoding.
        padding_len = input_ids[0].eq(self.tokenizer.pad_token_id).sum()
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len]  
  
        ### 1. answer start and end positions classification ###   
        logits = self.qa_outputs_1(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1) 
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1)
 
        ### 2. type classification, similar as class LongformerClassificationHead(nn.Module) https://huggingface.co/transformers/_modules/transformers/modeling_longformer.html#LongformerForSequenceClassification.forward ### 
        type_logits = self.linear_type_1(sequence_output[:,0]) 
        
        ### 3. supporting paragraph classification ###  
        sp_para_output = sequence_output[:,para_indexes,:]  
        sp_para_output_t = self.fnn_sp_para_1(sp_para_output)    

         # linear_sp_sent generates a single score for each sentence, instead of 2 scores for yes and no.   
        # Argument the score with additional score=0. The same way did in the HOTPOTqa paper
        sp_para_output_aux = torch.zeros(sp_para_output_t.shape, dtype=torch.float, device=sp_para_output_t.device) 
        predict_support_para = torch.cat([sp_para_output_aux, sp_para_output_t], dim=-1).contiguous() 
 
        ### 4. supporting fact classification ###     
        # the first sentence in a paragraph is leading by <p>, other sentences are leading by <s>
 
        sp_sent_output = sequence_output[:,sent_indexes,:]  
        sp_sent_output_t = self.fnn_sp_sent_1(sp_sent_output)     
        sp_sent_output_aux = torch.zeros(sp_sent_output_t.shape, dtype=torch.float, device=sp_sent_output_t.device) 
        predict_support_sent = torch.cat([sp_sent_output_aux, sp_sent_output_t], dim=-1).contiguous() 
        answer_loss_1, type_loss_1, sp_para_loss_1, sp_sent_loss_1  = self.loss_computation(start_positions, end_positions, start_logits, end_logits, q_type, type_logits, sp_para, predict_support_para, sp_sent, predict_support_sent)
 
        
        ################## prepare stage2: keep up to 5 paragraphs whose raw score is higher than a pre-specified threshold (-3.0) ##############
        para_stage2_indices = sp_para_output_t.flatten().topk(k=min(5, sp_para_output_t.numel()), dim=-1).indices     
        i = para_stage2_indices.numel() - 1   # the lowest score
        while i >= 0:
            if(sp_para_output_t.flatten()[para_stage2_indices[i]] > -3.0):
                para_stage2_indices = para_stage2_indices[:i+1]  # slice at the last score that is > -3.0
                break
            else:
                i -= 1
        if(i == -1):
            para_stage2_indices = sp_para_output_t.flatten().topk(k=min(2, sp_para_output_t.numel()), dim=-1).indices
        para_stage2_indices = para_stage2_indices.sort().values
        
        # relocate sp      
        sp_para_stage2 = sp_para[:,para_stage2_indices]
        sp_para_map = dict(zip(range(para_stage2_indices.numel()), para_stage2_indices.tolist())) # used to map back to sp_para in decode 
         
        s_to_p_map = []   
        for s in sent_indexes:
            s_to_p = torch.where(torch.le(para_indexes, s))[0][-1]     # last para_index smaller or equal to s
            s_to_p_map.append(s_to_p.item())   # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7, 7, 8, 9]
        sp_sent_stage2 = []
        sp_sent_map = {}
        for idx, s_to_p in enumerate(s_to_p_map):
            if(s_to_p in para_stage2_indices):
                sp_sent_map[len(sp_sent_stage2)] = idx
                sp_sent_stage2.append(sp_sent[0][idx].item())   
                
        sp_sent_stage2 = torch.tensor([sp_sent_stage2]).type_as(sp_para_stage2) 
         
        
        # input_stage2_ids
        idx_to_get = list(range(para_indexes[0]))    # before the first para: question  
        for idx, p_i in enumerate(para_stage2_indices):
            if(p_i < para_indexes.numel()-1):
                idx_to_get.extend(range(para_indexes[p_i], para_indexes[p_i+1]))    # between the para begin to next para begin 
            else:
                idx_to_get.extend(range(para_indexes[p_i], input_ids.size(1)))     # last para
        input_stage2_ids = input_ids[:,idx_to_get]
         
        # reposition answer  
        for i , start_pos in enumerate(start_positions[0]):
            if(start_pos != -1):   
                if(start_pos in idx_to_get):
                    start_positions[0][i] = idx_to_get.index(start_pos)
                else:
                    start_positions[0][i] = -1
        for i , end_pos in enumerate(end_positions[0]):
            if(end_pos != -1):   
                if(end_pos in idx_to_get):
                    end_positions[0][i] = idx_to_get.index(end_pos)
                else:
                    end_positions[0][i] = -1
        

                
            
        ########################################################## stage 2 ############################################################################    
        # local attention everywhere
        attention_mask = torch.ones(input_stage2_ids.shape, dtype=torch.long, device=input_stage2_ids.device)
        
        # global attention for the cls and all question tokens
        if(question_end_index != self._get_special_index(input_stage2_ids, [QUESTION_END]) ):
            print("suppose to be same as stage 1's question_end_index")
            assert(question_end_index == self._get_special_index(input_stage2_ids, [QUESTION_END]))
        question_end_index = self._get_special_index(input_stage2_ids, [QUESTION_END])  
        attention_mask[:,:question_end_index[0].item()+1] = 2  # from <cls> until </q> 
        
        # global attention for the sentence and paragraph special tokens  
        sent_indexes = self._get_special_index(input_stage2_ids, [SENT_MARKER_END])
        attention_mask[:, sent_indexes] = 2
        
        para_indexes = self._get_special_index(input_stage2_ids, [TITLE_START])
        attention_mask[:, para_indexes] = 2       
         

        # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
        input_stage2_ids, attention_mask = pad_to_window_size(
            input_stage2_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

        sequence_output = self.model_2(
                input_stage2_ids,
                attention_mask=attention_mask)[0]            
            
        # The pretrained hotpotqa model wasn't trained with padding, so remove padding tokens
        # before computing loss and decoding.
        padding_len = input_stage2_ids[0].eq(self.tokenizer.pad_token_id).sum()
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len]              
        
        ################### layers on top of sequence_output ############## 
        
        ### 1. answer start and end positions classification ###   
        logits = self.qa_outputs_2(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1) 
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1)
 
        ### 2. type classification, similar as class LongformerClassificationHead(nn.Module) https://huggingface.co/transformers/_modules/transformers/modeling_longformer.html#LongformerForSequenceClassification.forward ### 
        type_logits = self.linear_type_2(sequence_output[:,0]) 
        
        ### 3. supporting paragraph classification ###  
        sp_para_output = sequence_output[:,para_indexes,:]  
        sp_para_output_t = self.fnn_sp_para_2(sp_para_output) 

         # linear_sp_sent generates a single score for each sentence, instead of 2 scores for yes and no.   
        # Argument the score with additional score=0. The same way did in the HOTPOTqa paper
        sp_para_output_aux = torch.zeros(sp_para_output_t.shape, dtype=torch.float, device=sp_para_output_t.device) 
        predict_support_para = torch.cat([sp_para_output_aux, sp_para_output_t], dim=-1).contiguous() 
 
        ### 4. supporting fact classification ###     
        # the first sentence in a paragraph is leading by <p>, other sentences are leading by <s>
 
        sp_sent_output = sequence_output[:,sent_indexes,:]  
        sp_sent_output_t = self.fnn_sp_sent_2(sp_sent_output)     
        sp_sent_output_aux = torch.zeros(sp_sent_output_t.shape, dtype=torch.float, device=sp_sent_output_t.device) 
        predict_support_sent = torch.cat([sp_sent_output_aux, sp_sent_output_t], dim=-1).contiguous() 
        
        outputs = (start_logits, end_logits, type_logits, sp_para_output_t, sp_sent_output_t, input_stage2_ids, sp_para_map, sp_sent_map)  
        answer_loss_2, type_loss_2, sp_para_loss_2, sp_sent_loss_2  = self.loss_computation(start_positions, end_positions, start_logits, end_logits, q_type, type_logits, sp_para_stage2, predict_support_para, sp_sent_stage2, predict_support_sent)
 
        answer_loss = answer_loss_1 + answer_loss_2
        type_loss = type_loss_1 + type_loss_2
        sp_para_loss = sp_para_loss_1 + sp_para_loss_2
        sp_sent_loss = sp_sent_loss_1 + sp_sent_loss_2
        
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
                
            answer_loss = (start_loss + end_loss) / 2 
        else:
            answer_loss = 0
            print("start_positions is not None or end_positions is not None, ", "start_positions: ", start_positions, "end_positions: ", end_positions)
            
        crossentropy = torch.nn.CrossEntropyLoss(ignore_index=-1)
        type_loss = crossentropy(type_logits, q_type)  
        
        crossentropy_average = torch.nn.CrossEntropyLoss(reduction = 'mean', ignore_index=-1)   
        if(sp_para.view(-1).size(0) > 0):
            sp_para_loss = crossentropy_average(predict_support_para.view(-1, 2), sp_para.view(-1))
        else:
            sp_para_loss = torch.tensor(0).type_as(type_loss)  # when raw_context is []
            
        if(sp_sent.view(-1).size(0) > 0):    
            sp_sent_loss = crossentropy_average(predict_support_sent.view(-1, 2), sp_sent.view(-1))      
        else:
            sp_sent_loss = torch.tensor(0).type_as(type_loss)  # when raw_context is []
 
            
        return answer_loss, type_loss, sp_para_loss, sp_sent_loss  


#     %%add_to hotpotqa    
    def _get_special_index(self, input_ids, special_tokens):
        assert(input_ids.size(0)==1) 
        mask = input_ids != input_ids # initilaize 
        for special_token in special_tokens:
            mask = torch.logical_or(mask, input_ids.eq(self.tokenizer.convert_tokens_to_ids(special_token))) 
 
        token_indices = torch.nonzero(mask, as_tuple=False)        
         
 
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
        masked_target = target * (1 - target_mask.long())                 # replace all -1 in target with 0， tensor([[447,   0,   0,   0, ...]])
    
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)     # tensor([[0.4382, 0.2340, 0.2340, 0.2340 ... ]]), padding logits are all replaced by logits[0] 
 
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')                      # padding logits are all replaced by -inf
 
        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)
 
        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
 
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)
        
        # compute the loss
        loss = -(log_score - log_norm) 
        
        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`: when computing start_loss and end_loss for question with the gold answer of yes/no 
        # when `target` is all `ignore_index`, loss is 0 
        loss = loss[~torch.isinf(loss)].sum()
#         loss = torch.tanh(loss)
#         print("final loss: " + str(loss)) 
        return loss  


# ##### configure_ddp

# In[ ]:



    # A hook to overwrite to define your own DDP(DistributedDataParallel) implementation init. 
    # The only requirement is that: 
    # 1. On a validation batch the call goes to model.validation_step.
    # 2. On a training batch the call goes to model.training_step.
    # 3. On a testing batch, the call goes to model.test_step
    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


# ##### **configure_optimizers**

# In[ ]:



    def configure_optimizers(self):
        # Set up optimizers and (optionally) learning rate schedulers
        def lr_lambda(current_step):
            if current_step < self.args.warmup:
                return float(current_step) / float(max(1, self.args.warmup))
            return max(0.0, float(self.args.steps - current_step) / float(max(1, self.args.steps - self.args.warmup)))

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
     
# ##### **training_step**

# In[ ]:



    # A hook
    def on_epoch_start(self):
        print("Start epoch ", self.current_epoch, end='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
        pass


# In[ ]:
    def on_epoch_end(self): 
        print("end epoch ", self.current_epoch, end='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))     
        pass    
    


    def training_step(self, batch, batch_nb):
        # do the forward pass and calculate the loss for a batch 
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para, qid, answer = batch 
        # print("size of input_ids: " + str(input_ids.size())) 
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss  = output[:4]
        # print("answer_loss: ", answer_loss)
        # print("type_loss: ", type_loss)
        # print("sp_para_loss: ", sp_para_loss)
        # print("sp_sent_loss: ", sp_sent_loss)
        
    #     loss  = answer_loss +  type_loss + sp_para_loss + sp_sent_loss
        loss = answer_loss + 5*type_loss + 10*sp_para_loss + 10*sp_sent_loss
    #     print("weighted loss: ", loss)
    #     print("self.trainer.optimizers[0].param_groups[0]['lr']: ", self.trainer.optimizers[0].param_groups[0]['lr'])
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']  # loss.new_zeros(1) is tensor([0.]), converting 'lr' to tensor' by adding it.  
        
        tensorboard_logs = {'loss': loss, 'train_answer_loss': answer_loss, 'train_type_loss': type_loss, 
                            'train_sp_para_loss': sp_para_loss, 'train_sp_sent_loss': sp_sent_loss, 
                            'lr': lr #,
                            # 'mem': torch.tensor(torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3).type_as(loss) 
        }
        return tensorboard_logs

 
    # # the function is called for each batch after every epoch is completed
    # def training_end(self, output): 
    #     # print("training_end at epoch: ", self.current_epoch)
    # #     print("len(outputs): ",len(outputs))
    # #     print("output: ",output)
    
    #     # one batch only has one example
    #     avg_loss = output['loss']    
    #     avg_answer_loss = output['train_answer_loss']  
    #     avg_type_loss = output['train_type_loss']    
    #     avg_sp_para_loss = output['train_sp_para_loss']   
    #     avg_sp_sent_loss = output['train_sp_sent_loss'] 
    #     avg_lr = output['lr']      
         
     
    #     if self.trainer.use_ddp:
    #         torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
    #         avg_loss /= self.trainer.world_size 
    #         torch.distributed.all_reduce(avg_answer_loss, op=torch.distributed.ReduceOp.SUM)
    #         avg_answer_loss /= self.trainer.world_size 
    #         torch.distributed.all_reduce(avg_type_loss, op=torch.distributed.ReduceOp.SUM)
    #         avg_type_loss /= self.trainer.world_size 
    #         torch.distributed.all_reduce(avg_sp_para_loss, op=torch.distributed.ReduceOp.SUM)
    #         avg_sp_para_loss /= self.trainer.world_size 
    #         torch.distributed.all_reduce(avg_sp_sent_loss, op=torch.distributed.ReduceOp.SUM)
    #         avg_sp_sent_loss /= self.trainer.world_size 
    #         torch.distributed.all_reduce(avg_lr, op=torch.distributed.ReduceOp.SUM)
    #         avg_lr /= self.trainer.world_size 
            
     
    #     tensorboard_logs = { #'avg_train_loss': avg_loss, 
    #             'avg_train_answer_loss': avg_answer_loss, 'avg_train_type_loss': avg_type_loss, 'avg_train_sp_para_loss': avg_sp_para_loss, 'avg_train_sp_sent_loss': avg_sp_sent_loss, 'lr': avg_lr
    #           }
    
    #     return {'loss': avg_loss, 'log': tensorboard_logs}

# ##### validation_step

# In[ ]:



   # When the validation_step is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.
    def validation_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para, qid, answer = batch

        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output, input_stage2_ids, sp_para_map, sp_sent_map = output 
        loss = answer_loss + 5*type_loss + 10*sp_para_loss + 10*sp_sent_loss
        # print("qid: " + str(qid))
        answers_pred, sp_sent_pred, sp_para_pred = self.decode(input_stage2_ids, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output, sp_para_map, sp_sent_map)
 
 
        if(len(answers_pred) != 1):
            print("len(answers_pred) != 1")
            assert(len(answers_pred) == 1)

        pre_answer_score = answers_pred[0]['score']  # (start_logit + end_logit + p_type_score) / 3
        pre_answer = _normalize_text(answers_pred[0]['text'])
#         print("pred answer_score: " + str(pre_answer_score))
#         print("pred answer_text: " + str(pre_answer)) 

        gold_answer = _normalize_text(answer)
        f1, prec, recall = self.f1_score(pre_answer, gold_answer)
        em = self.exact_match_score(pre_answer, gold_answer) 
        f1 = torch.tensor(f1).type_as(loss)
        prec = torch.tensor(prec).type_as(loss)
        recall = torch.tensor(recall).type_as(loss)
        em = torch.tensor(em).type_as(loss)
#         print("f1: " + str(f1))
#         print("prec: " + str(prec))
#         print("recall: " + str(recall))
#         print("em: " + str(em))  

        if(len(sp_sent_pred) > 0):
            sp_sent_em, sp_sent_precision, sp_sent_recall, sp_sent_f1 = self.sp_metrics(sp_sent_pred, torch.where(sp_sent.squeeze())[0].tolist())
            sp_sent_em = torch.tensor(sp_sent_em).type_as(loss)
            sp_sent_precision = torch.tensor(sp_sent_precision).type_as(loss)
            sp_sent_recall = torch.tensor(sp_sent_recall).type_as(loss)
            sp_sent_f1 = torch.tensor(sp_sent_f1).type_as(loss)

   #         print("sp_sent_em: " + str(sp_sent_em))
   #         print("sp_sent_precision: " + str(sp_sent_precision))
   #         print("sp_sent_recall: " + str(sp_sent_recall))    
   #         print("sp_sent_f1: " + str(sp_sent_f1))    


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

        if(len(sp_para_pred) > 0): 
            sp_para_em, sp_para_precision, sp_para_recall, sp_para_f1 = self.sp_metrics(sp_para_pred, torch.where(sp_para.squeeze())[0].tolist())
            sp_para_em = torch.tensor(sp_para_em).type_as(loss)
            sp_para_precision = torch.tensor(sp_para_precision).type_as(loss)
            sp_para_recall = torch.tensor(sp_para_recall).type_as(loss)
            sp_para_f1 = torch.tensor(sp_para_f1).type_as(loss)
        else:
            sp_para_em, sp_para_precision, sp_para_recall, sp_para_f1 = torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss)
            

        return { 'vloss': loss, 'answer_loss': answer_loss, 'type_loss': type_loss, 'sp_para_loss': sp_para_loss, 'sp_sent_loss': sp_sent_loss,
                   'answer_score': pre_answer_score, 'f1': f1, 'prec':prec, 'recall':recall, 'em': em,
                   'sp_sent_em': sp_sent_em, 'sp_sent_f1': sp_sent_f1, 'sp_sent_precision': sp_sent_precision, 'sp_sent_recall': sp_sent_recall,
                   'sp_para_em': sp_para_em, 'sp_para_f1': sp_para_f1, 'sp_para_precision': sp_para_precision, 'sp_para_recall': sp_para_recall,
                   'joint_em': joint_em, 'joint_f1': joint_f1, 'joint_prec': joint_prec, 'joint_recall': joint_recall}


# ###### decode

# In[ ]:



    def decode(self, input_ids, start_logits, end_logits, type_logits, sp_para_logits, sp_sent_logits, sp_para_map, sp_sent_map):
#         print("decode")

        question_end_index = self._get_special_index(input_ids, [QUESTION_END])
    #     print("question_end_index: ", question_end_index)

        # one example per batch
        start_logits = start_logits.squeeze()
        end_logits = end_logits.squeeze()
    #     print("start_logits: ", start_logits)
    #     print("end_logits: ", end_logits)
        start_logits_indices = start_logits.topk(k=min(self.args.n_best_size, start_logits.size(0)), dim=-1).indices
        end_logits_indices = end_logits.topk(k=min(self.args.n_best_size, end_logits.size(0)), dim=-1).indices
        if(len(start_logits_indices.size()) > 1):
            print("len(start_logits_indices.size()): ", len(start_logits_indices.size()))
            assert("len(start_logits_indices.size()) > 1")
        p_type = torch.argmax(type_logits, dim=1).item()
        p_type_score = torch.max(type_logits, dim=1)[0] 
    #     print("type_logits: ", type_logits)
#         print("p_type: ", p_type)
#         print("p_type_score: ", p_type_score)

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
#             print("sorted_answers: " + str(sorted_answers))

            if len(sorted_answers) == 0:
                answers.append({'text': 'NoAnswerFound', 'score': -1000000, 'start_logit': -1000000, 'end_logit': -1000000, 'p_type_score': p_type_score})
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[0, answer['start']: answer['end'] + 1]

                answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                # remove [/sent], <t> and </t>
                for special_token in [SENT_MARKER_END, TITLE_START, TITLE_END, self.tokenizer.sep_token]:   
                    try:    
                        if(answer_tokens[0] == special_token):  
                            answer['start_logit'] = -2000000    
                        elif(answer_tokens[-1] == special_token):   
                            answer['end_logit'] = -2000000  
                                
                        answer_tokens.remove(special_token) 
                    except: 
                        pass    


                text = self.tokenizer.convert_tokens_to_string(answer_tokens)
                score = (answer['start_logit'] + answer['end_logit'] + p_type_score) / 3
                # score = (torch.sigmoid(answer['start_logit']) + torch.sigmoid(answer['end_logit']) + torch.sigmoid(p_type_score)) / 3
                answers.append({'text': text, 'score': score, 'start_logit': answer['start_logit'], 'end_logit': answer['end_logit'], 'p_type_score': p_type_score})
    #             print("answers: " + str(answers))
        elif p_type == 1:   
            answers.append({'text': 'yes', 'score': p_type_score, 'start_logit': -1000000, 'end_logit': -1000000, 'p_type_score': p_type_score})    
        elif p_type == 2:   
            answers.append({'text': 'no', 'score': p_type_score, 'start_logit': -1000000, 'end_logit': -1000000, 'p_type_score': p_type_score}) 
        else:
            assert False 


        sent_indexes = self._get_special_index(input_ids, [SENT_MARKER_END])
        para_indexes = self._get_special_index(input_ids, [TITLE_START])

        s_to_p_map = []   
        sp_sent_pred = []
        sp_para_pred = [] 
        for s in sent_indexes:
            s_to_p = torch.where(torch.le(para_indexes, s))[0][-1]     # last para_index smaller or equal to s
            s_to_p_map.append(s_to_p.item())  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7, 7, 8, 9]
        if(len(s_to_p_map)>0):      # https://arxiv.org/pdf/2004.06753.pdf section 3.3
            para_sent_logits_sum = torch.tensor([], device=sp_sent_logits.device)  
            evidence_candidates = {}
            para_sents_offset = [0]
            for i in range(s_to_p_map[-1]+1):
                para_sent_logits = torch.masked_select(sp_sent_logits.squeeze(), torch.tensor([p==i for p in s_to_p_map])) 
               
                para_sent_logits_sum = torch.cat([para_sent_logits_sum, torch.sum(para_sent_logits).unsqueeze(0) ]) 
                para_sents_offset.append(para_sent_logits.numel()+para_sents_offset[-1])  # [0, 21, 22, 24, 25, 26, 29, 30, 34, 35, 36], one more elements than num of paras   
                evidence_candidates[i] = torch.gt(para_sent_logits, 0.1).nonzero(as_tuple=True)[0]  # 0.1 is the threshold to be a candidate sentences
                 
            # para_sent_logits_sum: tensor([ 7.8180e-01,  6.8700e-02,  1.6170e-01,  7.4000e-02,  6.0000e-04,  2.2680e-01, -3.0400e-02,  9.3400e-02,  1.1200e-01,  1.2470e-01])
            # evidence_candidates: sentences with logits larger than threshold in each para,  [tensor([ 1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 16, 17, 19, 20]), tensor([0]), tensor([0, 1]), tensor([0]), tensor([0]), tensor([0, 1, 2]), tensor([], dtype=torch.int64), tensor([0, 2]), tensor([0]), tensor([0])]
            sp_para_pred = para_sent_logits_sum.squeeze().topk(k=min(para_sent_logits_sum.numel(), 2)).indices  # sp are from <=2 paragraphs
             
            sp_sent_pred = []
            if(sp_para_pred.numel() > 1):
                for para_idx in sp_para_pred: 
                    if(para_idx.item() in evidence_candidates):
                        sp_sent_pred.extend([(para_sents_offset[para_idx]+sent).item() for sent in evidence_candidates[para_idx.item()]]) 
                for idx, sp_para in enumerate(sp_para_pred):
                    sp_para_pred[idx] = sp_para_map[sp_para.item()]
            elif(sp_para_pred.numel()==1 and sp_para_pred.item() in evidence_candidates):
                sp_sent_pred = [(para_sents_offset[sp_para_pred]+sent).item() for sent in evidence_candidates[sp_para_pred.item()]]
                sp_para_pred = [sp_para_map[sp_para_pred.item()]]
 
            for idx, sp_sent in enumerate(sp_sent_pred):
                sp_sent_pred[idx] = sp_sent_map[sp_sent]

            sp_para_pred = sp_para_pred.tolist()
 
        return (answers, sp_sent_pred, sp_para_pred)

# ###### metrics

# In[ ]:



    def f1_score(self, prediction, ground_truth):
        normalized_prediction = _normalize_text(prediction)
        normalized_ground_truth = _normalize_text(ground_truth)
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
        return int(_normalize_text(prediction) == _normalize_text(ground_truth))


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


# ##### validation_epoch_end

# In[ ]:

    def validation_epoch_end(self, outputs):
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
        sp_sent_f1_scores = [x['sp_sent_f1'] for x in outputs]   
        sp_sent_em_scores = [x['sp_sent_em'] for x in outputs]   
        sp_sent_prec_scores = [x['sp_sent_precision'] for x in outputs]   
        sp_sent_recall_scores = [x['sp_sent_recall'] for x in outputs]   
        sp_para_f1_scores = [x['sp_para_f1'] for x in outputs]   
        sp_para_em_scores = [x['sp_para_em'] for x in outputs]   
        sp_para_prec_scores = [x['sp_para_precision'] for x in outputs]   
        sp_para_recall_scores = [x['sp_para_recall'] for x in outputs]   
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
     
            answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.float)
            prec_scores = self.sync_list_across_gpus(prec_scores, avg_loss.device, torch.float)
            recall_scores = self.sync_list_across_gpus(recall_scores, avg_loss.device, torch.float)
            
            sp_sent_f1_scores = self.sync_list_across_gpus(sp_sent_f1_scores, avg_loss.device, torch.float)
            sp_sent_em_scores = self.sync_list_across_gpus(sp_sent_em_scores, avg_loss.device, torch.float)
            sp_sent_prec_scores = self.sync_list_across_gpus(sp_sent_prec_scores, avg_loss.device, torch.float)
            sp_sent_recall_scores = self.sync_list_across_gpus(sp_sent_recall_scores, avg_loss.device, torch.float)
            
            sp_para_f1_scores = self.sync_list_across_gpus(sp_para_f1_scores, avg_loss.device, torch.float)
            sp_para_em_scores = self.sync_list_across_gpus(sp_para_em_scores, avg_loss.device, torch.float)
            sp_para_prec_scores = self.sync_list_across_gpus(sp_para_prec_scores, avg_loss.device, torch.float)
            sp_para_recall_scores = self.sync_list_across_gpus(sp_para_recall_scores, avg_loss.device, torch.float)
            
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
        avg_val_sp_para_f1 = sum(sp_para_f1_scores) / len(sp_para_f1_scores)  
        avg_val_sp_para_em = sum(sp_para_em_scores) / len(sp_para_em_scores)  
        avg_val_sp_para_prec = sum(sp_para_prec_scores) / len(sp_para_prec_scores)  
        avg_val_sp_para_recall = sum(sp_para_recall_scores) / len(sp_para_recall_scores)   
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
        print("avg_val_sp_para_f1: ", avg_val_sp_para_f1, end = '\t')   
        print("avg_val_sp_para_em: " , avg_val_sp_para_em, end = '\t')  
        print("avg_val_sp_para_prec: ", avg_val_sp_para_prec, end = '\t')   
        print("avg_val_sp_para_recall: ", avg_val_sp_para_recall)   
        print("avg_val_joint_f1: " , avg_val_joint_f1, end = '\t')  
        print("avg_val_joint_em: ", avg_val_joint_em, end = '\t')   
        print("avg_val_joint_prec: ", avg_val_joint_prec, end = '\t')   
        print("avg_val_joint_recall: ", avg_val_joint_recall)   
            
       
        logs = {'avg_val_loss': avg_loss, 'avg_val_answer_loss': avg_answer_loss, 'avg_val_type_loss': avg_type_loss, 
                'avg_val_sp_para_loss': avg_sp_para_loss, 'avg_val_sp_sent_loss': avg_sp_sent_loss,   
                'avg_val_f1': avg_val_f1 , 'avg_val_em': avg_val_em,  'avg_val_prec': avg_val_prec, 'avg_val_recall': avg_val_recall ,    
                'avg_val_sp_sent_f1': avg_val_sp_sent_f1, 'avg_val_sp_sent_em': avg_val_sp_sent_em,  'avg_val_sp_sent_prec': avg_val_sp_sent_prec, 'avg_val_sp_sent_recall': avg_val_sp_sent_recall, 
                'avg_val_sp_para_f1': avg_val_sp_para_f1, 'avg_val_sp_para_em': avg_val_sp_para_em,  'avg_val_sp_para_prec': avg_val_sp_para_prec, 'avg_val_sp_para_recall': avg_val_sp_para_recall, 
                'avg_val_joint_f1': avg_val_joint_f1, 'avg_val_joint_em': avg_val_joint_em,  'avg_val_joint_prec': avg_val_joint_prec, 'avg_val_joint_recall': avg_val_joint_recall 
        }   
       
       
        return logs
 
    def sync_list_across_gpus(self, list_to_sync, device, dtype):
        l_tensor = torch.tensor(list_to_sync, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()
    
     

# ##### test_step

# In[75]:



    def test_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para, qid, answer = batch

        print("test_step of qid: ", qid, end="\t") 
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, q_type, sp_sent, sp_para)
        answer_loss, type_loss, sp_para_loss, sp_sent_loss, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output, input_stage2_ids, sp_para_map, sp_sent_map = output 
        loss = answer_loss + 5*type_loss + 10*sp_para_loss + 10*sp_sent_loss
 
        answers_pred, sp_sent_pred, sp_para_pred = self.decode(input_stage2_ids, start_logits, end_logits, type_logits, sp_para_output, sp_sent_output, sp_para_map, sp_sent_map)
 
        if(len(answers_pred) != 1):
            print("len(answers_pred) != 1")
            assert(len(answers_pred) == 1)

        pre_answer_score = answers_pred[0]['score']  # (start_logit + end_logit + p_type_score) / 3
        pre_answer = _normalize_text(answers_pred[0]['text'])
        start_logit = answers_pred[0]['start_logit']    
        end_logit = answers_pred[0]['end_logit']    
        type_score = answers_pred[0]['p_type_score']
        
        gold_answer = _normalize_text(answer)
        f1, prec, recall = self.f1_score(pre_answer, gold_answer)
        em = self.exact_match_score(pre_answer, gold_answer) 
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

   #         print("sp_sent_em: " + str(sp_sent_em))
   #         print("sp_sent_precision: " + str(sp_sent_precision))
   #         print("sp_sent_recall: " + str(sp_sent_recall))    
   #         print("sp_sent_f1: " + str(sp_sent_f1))    

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

        
        if(len(sp_para_pred) > 0):
            sp_para_em, sp_para_precision, sp_para_recall, sp_para_f1 = self.sp_metrics(sp_para_pred, torch.where(sp_para.squeeze())[0].tolist())
            sp_para_em = torch.tensor(sp_para_em).type_as(loss)
            sp_para_precision = torch.tensor(sp_para_precision).type_as(loss)
            sp_para_recall = torch.tensor(sp_para_recall).type_as(loss)
            sp_para_f1 = torch.tensor(sp_para_f1).type_as(loss)
        else:
            sp_para_em, sp_para_precision, sp_para_recall, sp_para_f1 = torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss), torch.tensor(0.0).type_as(loss)
            
        
        
        
        self.logger.log_metrics({'answer_loss': answer_loss, 'type_loss': type_loss, 'sp_para_loss': sp_para_loss, 'sp_sent_loss': sp_sent_loss,    
                                    'answer_score': pre_answer_score, 'start_logit': start_logit, 'end_logit': end_logit,   
                                    'type_score': type_score,   
                                    'f1': f1, 'prec':prec, 'recall':recall, 'em': em,
                                    'sp_sent_f1': sp_sent_f1, 'sp_sent_precision': sp_sent_precision, 'sp_sent_recall': sp_sent_recall,  'sp_sent_em': sp_sent_em, 
                                    'sp_para_f1': sp_para_f1, 'sp_para_precision': sp_para_precision, 'sp_para_recall': sp_para_recall,  'sp_para_em': sp_para_em, 
                                    'joint_f1': joint_f1, 'joint_prec': joint_prec, 'joint_recall': joint_recall, 'joint_em': joint_em
                                }) 

        # print("pre_answer:\t", pre_answer, "\tgold_answer:\t", gold_answer, "\tstart_logits:\t", start_logits.cpu(), "\tend_logits:\t", end_logits.cpu(), "\ttype_logits:\t", type_logits.cpu()) 
        print("pre_answer:\t", pre_answer, "\tgold_answer:\t", gold_answer)

        return { 'vloss': loss, 'answer_loss': answer_loss, 'type_loss': type_loss, 'sp_para_loss': sp_para_loss, 'sp_sent_loss': sp_sent_loss,
                   'answer_score': pre_answer_score, 'f1': f1, 'prec':prec, 'recall':recall, 'em': em,
                    'sp_sent_f1': sp_sent_f1, 'sp_sent_precision': sp_sent_precision, 'sp_sent_recall': sp_sent_recall,  'sp_sent_em': sp_sent_em, 
                    'sp_para_f1': sp_para_f1, 'sp_para_precision': sp_para_precision, 'sp_para_recall': sp_para_recall,  'sp_para_em': sp_para_em, 
                   'joint_em': joint_em, 'joint_f1': joint_f1, 'joint_prec': joint_prec, 'joint_recall': joint_recall}


# ##### test_epoch_end

# In[173]:



    def test_epoch_end(self, outputs):
        print("test_epoch_end")
        avg_loss = torch.stack([x['vloss'] for x in outputs]).mean()  
        avg_answer_loss = torch.stack([x['answer_loss'] for x in outputs]).mean()  
        avg_type_loss = torch.stack([x['type_loss'] for x in outputs]).mean()  
        avg_sp_para_loss = torch.stack([x['sp_para_loss'] for x in outputs]).mean()  
        avg_sp_sent_loss = torch.stack([x['sp_sent_loss'] for x in outputs]).mean()  
             
        answer_scores = [x['answer_score'] for x in outputs]  # [item for sublist in outputs for item in sublist['answer_score']] #torch.stack([x['answer_score'] for x in outputs]).mean() # 
         
        f1_scores = [x['f1'] for x in outputs]  
        em_scores = [x['em'] for x in outputs]  
        prec_scores =  [x['prec'] for x in outputs] 
        recall_scores = [x['recall'] for x in outputs]  
        sp_sent_f1_scores = [x['sp_sent_f1'] for x in outputs]   
        sp_sent_em_scores = [x['sp_sent_em'] for x in outputs]   
        sp_sent_prec_scores = [x['sp_sent_precision'] for x in outputs]   
        sp_sent_recall_scores = [x['sp_sent_recall'] for x in outputs]   
        sp_para_f1_scores = [x['sp_para_f1'] for x in outputs]   
        sp_para_em_scores = [x['sp_para_em'] for x in outputs]   
        sp_para_prec_scores = [x['sp_para_precision'] for x in outputs]   
        sp_para_recall_scores = [x['sp_para_recall'] for x in outputs]           
        joint_f1_scores = [x['joint_f1'] for x in outputs]  
        joint_em_scores = [x['joint_em'] for x in outputs]  
        joint_prec_scores = [x['joint_prec'] for x in outputs]  
        joint_recall_scores = [x['joint_recall'] for x in outputs]
      
        print(f'before sync --> sizes:  {len(answer_scores)}')
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

    #         int_qids = self.sync_list_across_gpus(int_qids, avg_loss.device, torch.int)
            answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.float)
            prec_scores = self.sync_list_across_gpus(prec_scores, avg_loss.device, torch.float)
            recall_scores = self.sync_list_across_gpus(recall_scores, avg_loss.device, torch.float)
            
            sp_sent_f1_scores = self.sync_list_across_gpus(sp_sent_f1_scores, avg_loss.device, torch.float)
            sp_sent_em_scores = self.sync_list_across_gpus(sp_sent_em_scores, avg_loss.device, torch.float)
            sp_sent_prec_scores = self.sync_list_across_gpus(sp_sent_prec_scores, avg_loss.device, torch.float)
            sp_sent_recall_scores = self.sync_list_across_gpus(sp_sent_recall_scores, avg_loss.device, torch.float)
            
            sp_para_f1_scores = self.sync_list_across_gpus(sp_para_f1_scores, avg_loss.device, torch.float)
            sp_para_em_scores = self.sync_list_across_gpus(sp_para_em_scores, avg_loss.device, torch.float)
            sp_para_prec_scores = self.sync_list_across_gpus(sp_para_prec_scores, avg_loss.device, torch.float)
            sp_para_recall_scores = self.sync_list_across_gpus(sp_para_recall_scores, avg_loss.device, torch.float)            
            
            joint_f1_scores = self.sync_list_across_gpus(joint_f1_scores, avg_loss.device, torch.float)
            joint_em_scores = self.sync_list_across_gpus(joint_em_scores, avg_loss.device, torch.float)
            joint_prec_scores = self.sync_list_across_gpus(joint_prec_scores, avg_loss.device, torch.float)
            joint_recall_scores = self.sync_list_across_gpus(joint_recall_scores, avg_loss.device, torch.float)
       
           
            
        print(f'after sync --> sizes: {len(answer_scores)}')
        avg_test_f1 = sum(f1_scores) / len(f1_scores) 
        avg_test_em =  sum(em_scores) / len(em_scores)     
        avg_test_prec =  sum(prec_scores) / len(prec_scores)   
        avg_test_recall =  sum(recall_scores) / len(recall_scores)  
        avg_test_sp_sent_f1 =  sum(sp_sent_f1_scores) / len(sp_sent_f1_scores)    
        avg_test_sp_sent_em =  sum(sp_sent_em_scores) / len(sp_sent_em_scores) 
        avg_test_sp_sent_prec =  sum(sp_sent_prec_scores) / len(sp_sent_prec_scores)     
        avg_test_sp_sent_recall =  sum(sp_sent_recall_scores) / len(sp_sent_recall_scores)  
        avg_test_sp_para_f1 =  sum(sp_para_f1_scores) / len(sp_para_f1_scores)    
        avg_test_sp_para_em =  sum(sp_para_em_scores) / len(sp_para_em_scores) 
        avg_test_sp_para_prec =  sum(sp_para_prec_scores) / len(sp_para_prec_scores)     
        avg_test_sp_para_recall =  sum(sp_para_recall_scores) / len(sp_para_recall_scores)  
        avg_test_joint_f1 =  sum(joint_f1_scores) / len(joint_f1_scores)    
        avg_test_joint_em = sum(joint_em_scores) / len(joint_em_scores)    
        avg_test_joint_prec = sum(joint_prec_scores) / len(joint_prec_scores)  
        avg_test_joint_recall = sum(joint_recall_scores) / len(joint_recall_scores)   
          
        logs = {'avg_test_loss': avg_loss, 'avg_test_answer_loss': avg_answer_loss, 'avg_test_type_loss': avg_type_loss, 'avg_test_sp_para_loss': avg_sp_para_loss, 'avg_test_sp_sent_loss': avg_sp_sent_loss,   
                'avg_test_f1': avg_test_f1 , 'avg_test_em': avg_test_em,  'avg_test_prec': avg_test_prec, 'avg_test_recall': avg_test_recall ,    
                'avg_test_sp_sent_f1': avg_test_sp_sent_f1, 'avg_test_sp_sent_em': avg_test_sp_sent_em,  'avg_test_sp_sent_prec': avg_test_sp_sent_prec, 'avg_test_sp_sent_recall': avg_test_sp_sent_recall,    
                'avg_test_sp_para_f1': avg_test_sp_para_f1, 'avg_test_sp_para_em': avg_test_sp_para_em,  'avg_test_sp_para_prec': avg_test_sp_para_prec, 'avg_test_sp_para_recall': avg_test_sp_para_recall,    
                'avg_test_joint_f1': avg_test_joint_f1, 'avg_test_joint_em': avg_test_joint_em,  'avg_test_joint_prec': avg_test_joint_prec, 'avg_test_joint_recall': avg_test_joint_recall 
        }   
        return {'avg_test_loss': avg_loss, 'log': logs}

# ##### add_model_specific_args

# In[ ]:



    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='hotpotqa-longformer')  
        parser.add_argument("--save_prefix", type=str, required=True)   
        parser.add_argument("--train_dataset", type=str, required=False, help="Path to the training squad-format")  
        parser.add_argument("--dev_dataset", type=str, required=True, help="Path to the dev squad-format")  
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--gpus", type=str, default='0', help="ids of gpus. Default is gpu 0. To use CPU, use --gpus "" ")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00005, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="How often within one training epoch to check the validation set.")
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
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        return parser


# ### main

# In[ ]:


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) 

    if not args.test:     # if it needs to train, remove exsiting folder
        import shutil
        save_folder = os.path.join(args.save_dir, args.save_prefix)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder, ignore_errors=True)  #delete non-empty folder
    
    # In[ ]:
    
    print("Start at:", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
    
    hotpotqa.__abstractmethods__=set()   # without this, got an error "Can't instantiate abstract class hotpotqa with abstract methods" if these two abstract methods are not implemented in the same cell where class hotpotqa defined 
    model = hotpotqa(args)
    if torch.cuda.is_available():
        model.to('cuda')    # this is necessary to use gpu
    
    
    # In[ ]:

    logger = TestTubeLogger( # The TestTubeLogger adds a nicer folder structure to manage experiments and snapshots all hyperparameters you pass to a LightningModule.
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )
    
    
    # In[ ]:
    
    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_f1',
        mode='max',
        period=-1,
        prefix=''
    )
    
    
    # In[ ]:
       
    with open(args.train_dataset, "r", encoding='utf-8') as f: 
        train_set_size = len(json.load(f))  
    train_set_size = train_set_size * args.train_percent    # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus != "" else None
    num_devices = max(len(gpu_ids) , 1 )
    
    args.steps = args.epochs * train_set_size / (args.batch_size * num_devices)
    
    print(f'>>>>>>> #train_set_size: {train_set_size}, #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * num_devices} <<<<<<<')
    
    # In[ ]:
    
    
    trainer = pl.Trainer(gpus=-1, distributed_backend='ddp' if gpu_ids and (len(gpu_ids) > 1) else None,
                         track_grad_norm=-1, max_epochs=args.epochs, early_stop_callback=None, replace_sampler_ddp=False,
                         accumulate_grad_batches=args.batch_size,
                         train_percent_check = args.train_percent,
                         val_check_interval=args.val_every,
                         num_sanity_val_steps=2,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=args.no_progress_bar,
                         use_amp=not args.fp32 and gpu_ids and (len(gpu_ids) > 1), amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         check_val_every_n_epoch=1
                         )
     
    
    
    # In[ ]:
    
    
    if not args.test:
        trainer.fit(model)
    
    
    # In[ ]:
    
    print("Start Test at", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
    
    trainer.test(model)
    
    print("End at", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
    
    # In[ ]:
    

import argparse 
if __name__ == "__main__":  
    main_arg_parser = argparse.ArgumentParser(description="hotpotqa")   
    parser = hotpotqa.add_model_specific_args(main_arg_parser, os.getcwd()) 
    args = parser.parse_args()  
    for arg in vars(args):  
        print((arg, getattr(args, arg)))    
    main(args)