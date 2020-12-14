
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

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
EXTRA_ANSWERS = " yes no null </s>"

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
    

from matplotlib.cbook import flatten 
import spacy   
import en_core_web_lg                         
nlp = en_core_web_lg.load() 
#!python -m pip install pytextrank
# Fan: make 3 changes in pytextrank.py 
# 1. phrase_text = ' '.join(key[0] for key in phrase_key) 
#  p.text are the joint of lemma tokens with pos_ in kept_pos, and maintain the order when join    
# 2. add argumrnt 'chunk_type' to only consider named entity ('ner') or noun_chunks ('noun'), besides the default ('both') 
# 3. replace token.lemma_ with token.lemma_.lower().strip()
import pytextrank
tr = pytextrank.TextRank(pos_kept=["ADJ", "NOUN", "PROPN", "VERB", "NUM", "ADV"], chunk_type='both')  
nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
print(nlp.pipeline)   
# import neuralcoref
# neuralcoref.add_to_pipe(nlp)
 
from matplotlib.cbook import flatten

#!conda install networkx --yes
import networkx as nx
import itertools 


def create_para_graph(paras_phrases):
    G = nx.Graph()    
    top_para_phrases = []                     # node of the first (top ranked) phrases from each para 
    for para_phrases in paras_phrases:        # each para
        top_sent_phrases = []                 # node of the first (top ranked) phrases from each sent 
        for sent_phrases in para_phrases:     # each sent
            
            # complete graph for each sent
            sent_G = nx.Graph()
            sent_G.add_nodes_from([phrase[0] for phrase in sent_phrases])  
            sent_G.add_edges_from(itertools.combinations([phrase[0] for phrase in sent_phrases], 2)) 
            G = nx.compose(G, sent_G)         # union of the node sets and edge sets
            
            
            # add an edge between the top ranked phrases from each sent to bridge sents
            if(sent_phrases):
                for top_sent_phrase in top_sent_phrases:
                    G.add_edge(top_sent_phrase[0], sent_phrases[0][0])  # sent_phrases[0] is the top ranked phrase of the sentence
                top_sent_phrases.append(sent_phrases[0])     
            
        top_sent_phrases = sorted(top_sent_phrases, key=lambda x: x[1], reverse=True)      # x[0]: phrase text,  x[1]: phrase rank
        
        
        # add an edge between the top ranked phrases from each para to bridge paras
        if(top_sent_phrases):
            for top_para_phrase in top_para_phrases: 
                G.add_edge(top_para_phrase[0], top_sent_phrases[0][0])  # top_sent_phrases[0] is the top ranked phrase of current para
            top_para_phrases.append(top_sent_phrases[0])
     
    # Draw
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(20,10))
#     nx.draw(G, pos, with_labels=True, edge_color='black', width=1, linewidths=1,
#             node_size=500, node_color='orange', alpha=0.9                           
#             )     
    return G

import re
import string

from networkx.algorithms import approximation as approx
def reduce_context_with_phares_graph(json_dict, outfile, gold_paras_only=False):
    """function to compute reduced context with phrase graph.

    Args:
        json_dict: The original data load from hotpotqa file.
        gold_paras_only: when is true, only use the 2 paragraphs that contain the gold supporting facts; if false, use all the 10 paragraphs
 
    Returns:
        a new file save additional phrase-related info and the reduced context

    """
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    new_dict = {"data": []} 
    common_phrases_num_le2 = 0
    extended = 0
    answer_in_reduced_context = 0
    answer_in_context = 0
    reduced_context_ratios = []
    for e_id, example in enumerate(json_dict): 

        support_para = set(
            para_title for para_title, _ in example["supporting_facts"]
        )
        sp_set = set(list(map(tuple, example['supporting_facts'])))
        
        raw_contexts = example["context"]
#         if gold_paras_only: 
#        raw_contexts = [lst for lst in raw_contexts if lst[0] in support_para]    
        is_supporting_para = []  # a boolean list with 10 True/False elements, one for each paragraph
        is_sup_fact = []         # a boolean list with True/False elements, one for each context sentence
        paras_phrases = []                                                # phrases of all 10 paragraghs
        for i, para_context in enumerate(raw_contexts):                   # each para
            is_supporting_para.append(para_context[0] in support_para)   
            for sent_id, sent in enumerate(para_context[1]):
                is_sup_fact.append( (para_context[0], sent_id) in sp_set )  
 
            para_context[0] = normalize_answer(para_context[0])
            para_context[1] = [normalize_answer(sent) for sent in para_context[1]]

            sent_docs = list(nlp.pipe([para_context[0]] + para_context[1]))   
            para_phrases = []                                        
            for sent_doc in sent_docs:                                    # each sent in a para
                sent_phrases = [(p.text, p.rank) for p in sent_doc._.phrases if(p.text != '')]  # phrases from each sentence 
                para_phrases.append(sent_phrases)       
            paras_phrases.append(para_phrases)    

        contexts = [TITLE_START + ' ' + lst[0]  + ' ' + TITLE_END + ' ' + (' ' + SENT_MARKER_END +' ').join(lst[1]) + ' ' + SENT_MARKER_END for lst in raw_contexts]  
        context = " ".join(contexts)                                                     
        answer = normalize_answer(example["answer"])  
        
        if (answer != '' and len(list(re.finditer(answer, context, re.IGNORECASE))) > 0):
            answer_in_context += 1
        
        paras_phrases_graph = create_para_graph(paras_phrases)
        
        question = normalize_answer(example["question"])
        question_doc = nlp(question)
        question_phrases = [(p.text, p.rank) for p in question_doc._.phrases if(p.text != '')] 
        question_phrases_text = [p[0] for p in question_phrases]
        
        all_sent_phrases_text =  list(flatten(paras_phrases))[::2]        # every other element is text, others are rank. 
        common_phrases = list(set(all_sent_phrases_text).intersection(question_phrases_text)) 
        question_only_phrase = list(set(question_phrases_text).difference(common_phrases)) 
        
        # print("question_phrases_text: ", question_phrases_text)
        # print("common_phrases: ", common_phrases)
#         print("question_only_phrase: ", question_only_phrase)
        
        if(len(common_phrases) > 1):
            common_phrases_num_le2 += 1
            path_phrases = list(approx.steinertree.steiner_tree(paras_phrases_graph, common_phrases).nodes)  # to find the shortest path cover all common_phrases  
            extended_phrases = path_phrases + question_only_phrase  
            if(len(extended_phrases) > len(question_phrases_text)):
                extended += 1
        else: #  0 or 1 common phrases
            path_phrases = common_phrases             
            extended_phrases = question_phrases_text
            
        # print("extended_phrases: ", extended_phrases)
         
        
#         example["question_phrases"] = question_phrases
#         example["paras_phrases"] = paras_phrases
#         example["common_phrases"] = common_phrases
#         example["path_phrases"] = path_phrases
#         example["extended_phrases"] = extended_phrases
#         print("context: ", context)    
#         print("\n\n") 
#         print("question_phrases: ", question_phrases)    
        # print("paras_phrases")
        # for paras_phrase in paras_phrases:
        #     print(paras_phrase)
        #     print("\n") 
#         print("all_sent_phrases_text: ", all_sent_phrases_text) 
#         print("\n\n") 
        
 
        raw_reduced_contexts = []     # sentences contain one of the phrases in the path 
        number_sentences = 0
        number_reduced_sentences = 0 
        for para_id, (para_title, para_lines) in enumerate(raw_contexts):
# #             print("para_id, para_title, para_lines",para_id, para_title, para_lines)
 
            number_sentences += len(para_lines)
            reduced_para = []
            for sent_id, sent in enumerate(para_lines):
 
                for phrase in path_phrases:
                    if(phrase in list(flatten(paras_phrases[para_id][sent_id]))[::2]):  # every other element is text, others are rank
                        reduced_para.append(sent)
                        number_reduced_sentences += 1 
                        break     # if current sentence contains a phrase in path, this sentence is added to reduced sentence, and no need to continue checking whether it contains other phrases
            if(len(reduced_para) > 0):
                raw_reduced_contexts.append([para_title, reduced_para])
        assert number_reduced_sentences <= number_sentences                    
        reduced_context_ratios.append(number_reduced_sentences / number_sentences)    
        
        reduced_contexts = [TITLE_START + ' ' + lst[0]  + ' ' + TITLE_END + ' ' + (' ' + SENT_MARKER_END +' ').join(lst[1]) + ' ' + SENT_MARKER_END for lst in raw_reduced_contexts]    
        reduced_context = " ".join(reduced_contexts)  
        
        if (answer != '' and len(list(re.finditer(answer, reduced_context, re.IGNORECASE))) > 0):
            answer_in_reduced_context += 1
        

        new_dict["data"].append(
            create_para_dict(
                create_example_dict(
                    context=reduced_context,
                    answer=answer,
                    id = example["_id"],
                    question=example["question"],
                    is_sup_fact = is_sup_fact,
                    is_supporting_para = is_supporting_para 
                )
            )
        )         
        
        # print("number_sentences: ", number_sentences)
        # print("number_reduced_sentences: ", number_reduced_sentences)

#         now = datetime.now()
#         current_time = now.strftime("%H:%M:%S")
#         print("Time =", current_time)
    print("number of questions with answer in context: ", answer_in_context)
    print("common_phrases_num_le2: ", common_phrases_num_le2) 
    print("number of questions with extended phrases: ", extended)
    print("number of questions with answer in reduced_context: ", answer_in_reduced_context)
    print("reduced context ratios: ", reduced_context_ratios)
    print("average ratio of reduced context: ", sum(reduced_context_ratios)/len(reduced_context_ratios))
    
    with open(outfile, 'w') as out_file:
        json.dump(new_dict, out_file)
    return  

def normalize_answer(s):

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
 
import json
import argparse 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, default='small.json')
    parser.add_argument("--outfile", type=str, default='small_out.json')
    args = parser.parse_args()
    with open(args.datafile, "r", encoding='utf-8') as f:  
        reduce_context_with_phares_graph(json.load(f), args.outfile)  


if __name__ == "__main__":
    main()

