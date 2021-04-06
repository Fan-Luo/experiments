
from datetime import datetime
 
import tqdm 
from datetime import datetime 
import pytz 
timeZ_Az = pytz.timezone('US/Mountain') 

# QUESTION_START = '[question]'
# QUESTION_END = '[/question]' 
# TITLE_START = '<t> , '  # indicating the start of the title of a paragraph (also used for loss over paragraphs)
# TITLE_END = ', </t> . '   # indicating the end of the title of a paragraph, add , to avoid tilte to be recognized as part of the first entity in the sentence after
SENT_MARKER_END = ', </sent> , '  # indicating the end of the title of a sentence (used for loss over sentences)
# PAR = '[/par]'  # used for indicating end of the regular context and beginning of `yes/no/null` answers
# EXTRA_ANSWERS = " yes no null </s>"

# def create_example_dict(context, answer, id, question, is_sup_fact, is_supporting_para):
#     return {
#         "context": context,
#         "qas": [                        # each context corresponds to only one qa in hotpotqa
#             {
#                 "answer": answer,
#                 "id": id,
#                 "question": question,
#                 "is_sup_fact": is_sup_fact,
#                 "is_supporting_para": is_supporting_para
#             }
#         ],
#     }

# def create_para_dict(example_dicts):
#     if type(example_dicts) == dict:
#         example_dicts = [example_dicts]   # each paragraph corresponds to only one [context, qas] in hotpotqa
#     return {"paragraphs": example_dicts} 
    

import spacy   
import en_core_web_lg          
nlp1 = en_core_web_lg.load() 
nlp2 = en_core_web_lg.load() 

from spacy.symbols import ORTH, LEMMA, POS
nlp1.tokenizer.add_special_case('</sent>', [{ ORTH: '</sent>', LEMMA: '</sent>', POS: 'SYM'}]) 
nlp1.tokenizer.add_special_case('</t>', [{ORTH: '</t>', LEMMA: '</t>', POS: 'SYM'}]) 
nlp1.tokenizer.add_special_case('<t>', [{ORTH: '<t>', LEMMA: '<t>', POS: 'SYM'}])  
import neuralcoref
neuralcoref.add_to_pipe(nlp1, greedyness=0.55) # between 0 and 1. The default value is 0.5.


#!python -m pip install pytextrank
# Fan: make 3 changes in pytextrank.py 
# 1. phrase_text = ' '.join(key[0] for key in phrase_key) 
#  p.text are the joint of lemma tokens with pos_ in kept_pos, and maintain the order when join    
# 2. add argumrnt 'chunk_type' to only consider named entity ('ner') or noun_chunks ('noun'), besides the default ('both') 
# 3. replace token.lemma_ with token.lemma_.lower().strip()
import pytextrank
tr = pytextrank.TextRank(pos_kept=["ADJ", "NOUN", "PROPN", "VERB", "NUM", "ADV"], chunk_type='both')  
nlp2.add_pipe(tr.PipelineComponent, name='textrank', last=True)



# %matplotlib inline
# import matplotlib.pyplot as plt
from matplotlib.cbook import flatten

#!conda install networkx --yes
import networkx as nx
import itertools 
from networkx.readwrite import json_graph
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from fuzzywuzzy import utils

def create_para_graph(paras_phrases):
    G = nx.Graph()     
    for pi, para_phrases in enumerate(paras_phrases):        # each para 
        for si, sent_phrases in enumerate(para_phrases):     # each sent
            
            # complete graph for each sent
            G.add_nodes_from([(phrase[0], {"score": phrase[1]}) for phrase in sent_phrases])  
            for node1, node2 in itertools.combinations([phrase[0] for phrase in sent_phrases], 2):
                if(G.has_edge(node1, node2)):
                    G[node1][node2]['src'].append((pi, si))
                else:
                    G.add_edge(node1, node2, src = [(pi, si)])
                                               
                                                
            # add edge between title phrases and first phrase of the sentence
            # si = 0, sent_phrases = para_phrases[0] are phrases from title 
            for phrase in para_phrases[0]:
                if(len(sent_phrases) > 0 and sent_phrases[0] != phrase):
                    if(G.has_edge(sent_phrases[0], phrase)):
                        G[sent_phrases[0]][phrase]['src'].append((pi, 'title', si))
                    else:
                        G.add_edge(sent_phrases[0], phrase, src = [(pi, 'title', si)]) 
     
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
    # new_dict = {"data": []} 
    data = []
 
    for e_id, example in enumerate(json_dict): 
        
        print("_id: ", example["_id"], end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
        
        raw_contexts = example["context"]
#         if gold_paras_only: 
#        raw_contexts = [lst for lst in raw_contexts if lst[0] in support_para]    
        paras_phrases = []                                                # phrases of all 10 paragraghs
        # contexts = []
 
        for i, para_context in enumerate(raw_contexts):                   # each para
        
            print("process paragraph ", i, end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
        
            title = _normalize_text(para_context[0])          
            sents = [_normalize_text(sent) for sent in para_context[1]]
            
            print("after normalize text", end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
            
            # resolve coreference
            num_sents_before_coref_resolved = len(sents)
            # print("numbe of sents before coref: ", num_sents_before_coref_resolved)
            sents_joint =  (' ' + SENT_MARKER_END +' ').join(sents) 
            sents_doc = nlp1(sents_joint)
            # print("resolved_sents: ", sents_doc._.coref_resolved) 
            sents_coref_resolved = sents_doc._.coref_resolved.split(SENT_MARKER_END)
            num_sents_after_coref_resolved = len(sents_coref_resolved)
            # print("numbe of sents after coref: ", num_sents_after_coref_resolved)
            
            if(num_sents_before_coref_resolved == num_sents_after_coref_resolved):
                sent_docs = list(nlp2.pipe([title] + sents_coref_resolved))       
            else:
                sent_docs = list(nlp2.pipe([title] + sents))
            
            print("after coreference resolution", end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
            
            
            para_phrases = []                                        
            for sent_doc in sent_docs:                                    # each sent in a para
                sent_phrases = [(p.text, p.rank) for p in sent_doc._.phrases if(p.text != '')]  # phrases from each sentence 
                para_phrases.append(sent_phrases)       
            paras_phrases.append(para_phrases)    
            
            print("after extract phrases", end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))

        all_sent_phrases_text =  list(flatten(paras_phrases))[::2]        # every other element is text, others are rank. 
        
        question = _normalize_text(example["question"])
        question_doc = nlp2(question)
        question_phrases = [(p.text, p.rank) for p in question_doc._.phrases if(p.text != '')] 
        question_phrases_text = [p[0] for p in question_phrases]
        # question_phrases_text = set(list(flatten([p.split() for p in question_phrases_text])) + question_phrases_text) # add phrase words

        print("after extracting phrases for the question", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))

        paras_phrases_graph = create_para_graph(paras_phrases)
        Subgraphs = [paras_phrases_graph.subgraph(c).copy() for c in nx.connected_components(paras_phrases_graph)]
        
        RG = nx.Graph()    # relevant components  
        represnetive_nodes = []  # more likely to include the represnetive_nodes in the final path       
        for S in Subgraphs:
            for phrase in question_phrases_text:
                if S.has_node(phrase):
                    RG = nx.compose(RG, S)  # joint the relevant components
                    represnetive_node = sorted(S.nodes.data('score'), key=lambda x: x[1], reverse=True)[0]  # node with highest score
                    represnetive_nodes.append(represnetive_node) 
                    break
        
        for node1, node2 in itertools.combinations([phrase[0] for phrase in represnetive_nodes], 2):  
            RG.add_edge(node1, node2, src = 'components')
            
            
        # map pharse similar to question phrases to question phrase, then fnd common phrases
        common_phrases = set()
        mapping = {}
        for phrase in RG.nodes:
            if(phrase in question_phrases_text):    # has a exact match
                common_phrases.add(phrase)
                continue
                
            # check partial match
            if (utils.full_process(phrase) and question_phrases_text != []):    # only exectute when query is valid. To avoid WARNING:root:Applied processor reduces input query to empty string, all comparisons will have score 0.
                simi_phrase, similarity = process.extractOne(phrase, question_phrases_text, scorer=fuzz.token_sort_ratio)  # most similar 
                if(similarity > 70): 
                    mapping[phrase] = simi_phrase   
                    common_phrases.add(simi_phrase)
                else:
                    simi_phrase, similarity = process.extractOne(phrase, question_phrases_text, scorer=fuzz.partial_ratio)    # match 'woman' and 'businesswoman'
                    if(similarity == 100):   # substring
                        mapping[phrase] = simi_phrase   
                        common_phrases.add(simi_phrase)
                 
                
        RG = nx.relabel_nodes(RG, mapping)      # match 'english government position' with 'government position'

        print("after graph construction", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
 
        question_only_phrase = list(set(question_phrases_text).difference(common_phrases))
        if(len(common_phrases) > 1):
            path_phrases = list(approx.steinertree.steiner_tree(RG, common_phrases).nodes)  # to find the shortest path cover all common_phrases  
            extended_phrases = path_phrases + question_only_phrase  
        else: #  0 or 1 common phrases
            path_phrases = list(common_phrases)             
            extended_phrases = question_phrases_text

        print("after path finding", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))

        P = RG.subgraph(extended_phrases)        
        # print(P.edges(data=True))
        path_data = json_graph.node_link_data(P)
        example["path"] = path_data
        example["question_phrases"] = question_phrases
        example["question_phrases_text"] = question_phrases_text 
        example["paras_phrases"] = paras_phrases
    #     example["all_sent_phrases_text"] = all_sent_phrases_text
        example["common_phrases"] = list(common_phrases)
        example["question_only_phrase"] = question_only_phrase
        example["path_phrases"] = path_phrases
        example["extended_phrases"] = extended_phrases 
        
        
        # example["relevant_graph_nodes"] = list(RG.nodes) 
        # print("extended_phrases: ", extended_phrases)
#         print("context: ", context)    
#         print("\n\n") 
#         print("question_phrases: ", question_phrases)    
        # print("paras_phrases")
        # for paras_phrase in paras_phrases:
        #     print(paras_phrase)
        #     print("\n") 
#         print("all_sent_phrases_text: ", all_sent_phrases_text) 
#         print("\n\n") 
        
 
        # construct the reduced_context    
        raw_reduced_contexts = []     # sentences contain one of the extended_phrases
        number_sentences = 0
        number_reduced_sentences = 0 
        kept_para_sent = []
        for para_id, (para_title, para_lines) in enumerate(raw_contexts): 
            number_sentences += len(para_lines)
            reduced_para = []
            kept_sent = []
            for sent_id, sent in enumerate(para_lines):
    
                for phrase in extended_phrases:
                    # fuzzy macth for construct reduce_conext
                    sentence_phrases = list(flatten(paras_phrases[para_id][sent_id+1]))[::2]  # paras_phrases[para_id][0] are phrases from the title, every other element is text, others are rank  
                                     
                    if(phrase in sentence_phrases):  # has a exact match, current sentence contains one of the extended_phrases
                        reduced_para.append(sent)
                        number_reduced_sentences += 1 
                        kept_sent.append(sent_id)
                        break     # no need to continue checking whether current sentence contains other extended_phrases
                    
                    # check partial match
                    if (utils.full_process(phrase) and sentence_phrases != []):    # only exectute when query is valid. To avoid WARNING:root:Applied processor reduces input query to empty string, all comparisons will have score 0.
                        simi_phrase, similarity = process.extractOne(phrase, sentence_phrases, scorer=fuzz.token_sort_ratio)  
                        if(similarity > 70): # current sentence contains at least one  phrase very similar to the extended_phrase
                            reduced_para.append(sent)
                            number_reduced_sentences += 1 
                            kept_sent.append(sent_id)
                            break 
                        else:
                            simi_phrase, similarity = process.extractOne(phrase, sentence_phrases, scorer=fuzz.partial_ratio)    # match 'woman' and 'businesswoman'
                            if(similarity == 100):   # current sentence contains substring of extended_phrase
                                reduced_para.append(sent)
                                number_reduced_sentences += 1 
                                kept_sent.append(sent_id)
                                break 
                            
                            
            if(len(reduced_para) > 0):
                raw_reduced_contexts.append([para_title, reduced_para])
                kept_para_sent.append(kept_sent)
            else:
                for phrase in extended_phrases:
                    if(phrase in list(flatten(paras_phrases[para_id][0]))[::2]):   # only tilte contains one of the extended_phrases
                        raw_reduced_contexts.append([para_title, []])
                        kept_para_sent.append(kept_sent)
                        break
                    
        print("after reconstruct reduced context", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))    
                    
        assert number_reduced_sentences <= number_sentences                     
        
     
        supporting_facts = []
        support_para = set(
            para_title for para_title, _ in example["supporting_facts"]
        )
        sp_set = set(list(map(tuple, example['supporting_facts'])))                       # a list of (title, sent_id in orignal context) 
        for i, para_reduced_context in enumerate(raw_reduced_contexts):                   # each para
            if(para_reduced_context[0] in support_para):
                for sent_id, orig_sent_id in enumerate(kept_para_sent[i]):
                    if( (para_reduced_context[0], orig_sent_id) in sp_set ):
                        supporting_facts.append([para_reduced_context[0], sent_id])

        print("after reconstruct reduced sp", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))    

        example['context'] = raw_reduced_contexts
        example['supporting_facts'] = supporting_facts
        example['kept_para_sent'] = kept_para_sent
        data.append(example)         
        
        
        # print("number_sentences: ", number_sentences)
        # print("number_reduced_sentences: ", number_reduced_sentences)
 
    # print("number of questions with at least 2 phrases shared by question and context: ", common_phrases_num_le2) 
    # print("number of questions with extended phrases from context besides question: ", extended) 
    # print("reduced context ratios: ", reduced_context_ratios)
    # print("average ratio of reduced context: ", sum(reduced_context_ratios)/len(reduced_context_ratios))
    
    with open(outfile, 'w') as out_file:
        json.dump(data, out_file)
        
    print("after saving to file", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))            

    return  

def _normalize_text(s):

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
    print("Start at", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
    
    main()
    
    print("End at", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
