
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
neuralcoref.add_to_pipe(nlp1, greedyness=0.53) # between 0 and 1. The default value is 0.5.


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
        
        question = basic_normalize(example["question"])
        question_doc = nlp2(question)
        question_phrases = [(_normalize_text(p.text), p.rank) for p in question_doc._.phrases if(p.text != '')] 
        question_phrases_text = [p[0] for p in question_phrases] 
        # question_phrases_text = set(list(flatten([p.split() for p in question_phrases_text])) + question_phrases_text) # add phrase words

        raw_contexts = example["context"]
#         if gold_paras_only: 
#        raw_contexts = [lst for lst in raw_contexts if lst[0] in support_para]    
        paras_phrases = []                                                # phrases of all 10 paragraghs
        # contexts = []
             
        for i, para_context in enumerate(raw_contexts):                   # each para
        
            print("process paragraph ", i, end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
        
            title = basic_normalize(para_context[0])    
            sents = [ basic_normalize(sent) for sent in para_context[1]]
            
            num_sents_before_coref_resolved = len(sents) 
            sents_joint =  (' ' + SENT_MARKER_END +' ').join(sents)
            sents_doc = nlp1(sents_joint) 
            sents_coref_resolved = sents_doc._.coref_resolved.split(SENT_MARKER_END)
            num_sents_after_coref_resolved = len(sents_coref_resolved) 
            
            if(num_sents_before_coref_resolved == num_sents_after_coref_resolved):
                sent_docs = list(nlp2.pipe([title] + sents_coref_resolved))       
            else:
                sent_docs = list(nlp2.pipe([title] + sents))
            
            print("after coreference resolution", end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))
            
            
            para_phrases = []                                        
            for sent_doc in sent_docs:                                    # each sent in a para
                sent_phrases = [(_normalize_text(p.text), p.rank) for p in sent_doc._.phrases if(p.text != '')]  # phrases from each sentence 
                para_phrases.append(sent_phrases)       
            paras_phrases.append(para_phrases)    
            
            print("after extract phrases", end ='\t')
            print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))

        # all_sent_phrases_text =  list(flatten(paras_phrases))[::2]        # every other element is text, others are rank. 
 

        print("after extracting phrases for the question", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))

        RG = create_relevant_graph(paras_phrases, question_phrases_text)
        RG, common_phrases, mapping = find_common_mapping(RG, question_phrases_text)# mapping matchs paras_phrases with question_phrases
        RG, dup_sets = dedup_nodes_in_graph(RG, question_phrases_text)              # dedup paras_phrases in RG for finding meanningful path

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

        # expand to include previously mapped nodes 
        reversed_mapping = {} 
        for k,v in mapping.items():
            if v in reversed_mapping:
                reversed_mapping[v].append(k)
            else:
                reversed_mapping[v] = [k] 
        for phrase in extended_phrases:        
            if phrase in reversed_mapping:
                extended_phrases.extend(reversed_mapping[phrase])
        
        # futher expand to include merged nodes, that is, also include phrases that from the same dup_set 
        extended_phrases_merged = set()
        for phrase in extended_phrases:
            idx_phrase = [idx for idx, dup_set in enumerate(dup_sets) if(phrase in dup_set)]   # idx_phrase[0]: idx of the set where phrase in
            if(len(idx_phrase) > 0):
                extended_phrases_merged = extended_phrases_merged | dup_sets[idx_phrase[0]]    # merge sets
                extended_phrases_merged.remove(phrase)
        extended_phrases.extend(list(extended_phrases_merged))
        introduced_phrases = list(set(extended_phrases) - set(question_phrases_text))

        raw_reduced_contexts, kept_para_sent = construct_reduced_context(raw_contexts, paras_phrases, extended_phrases, mapping) 
        reduced_supporting_facts, reduced_supporting_facts_in_original_id = construct_reduced_supporting_facts(example["supporting_facts"], raw_reduced_contexts, kept_para_sent) 
        print("after reconstruct reduced context and reduced sp", end ='\t')
        print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))    


        P = RG.subgraph(path_phrases)        
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
        example["introduced_phrases"] = introduced_phrases
        example['context'] = raw_reduced_contexts
        example['supporting_facts'] = reduced_supporting_facts
        example['reduced_supporting_facts_in_original_id'] = reduced_supporting_facts_in_original_id
        example['kept_para_sent'] = kept_para_sent
        data.append(example)         

    
    with open(outfile, 'w') as out_file:
        json.dump(data, out_file)
        
    print("after saving to file", end ='\t')
    print(datetime.now(timeZ_Az).strftime("%Y-%m-%d %H:%M:%S"))            

    return  

def create_relevant_graph(paras_phrases, question_phrases_text):
    G = create_para_graph(paras_phrases)
    Subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
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
        RG.add_edge(node1, node2, source = 'components')

    return RG


def find_common_mapping(G, question_phrases_text):
    # fuzzy macth for common phrases: map phrases similar to question phrases to question phrase, then find common phrases
    common_phrases = set()
    mapping = {}
    for phrase in G.nodes:
        if(phrase in question_phrases_text):    # has a exact match
            common_phrases.add(phrase)
            continue
            
        # check partial match
        inclusion_similar_phrase = inclusion_best_match(phrase, question_phrases_text)
        if(inclusion_similar_phrase): 
            mapping[phrase] = inclusion_similar_phrase   
            common_phrases.add(inclusion_similar_phrase)
    
    G = nx.relabel_nodes(G, mapping)      # match 'english government position' with 'government position'
    
    return G, common_phrases, mapping

def inclusion_best_match(query, choices):
    if (utils.full_process(query) and choices != []):  # only exectute when query is valid. To avoid WARNING:root:Applied processor reduces input query to empty string, all comparisons will have score 0.
        inclusion_phrases = [simi_phrase for (simi_phrase, similarity) in process.extractBests(query, choices, scorer=fuzz.token_set_ratio) if similarity ==100]  # match '1977 film' and '1977', but will not match substring 'woman' and 'businesswoman', avid nosiy such as 'music' and 'us'
        if(inclusion_phrases!= []):
            simi_phrase, similarity = process.extractOne(query, inclusion_phrases, scorer=fuzz.ratio) # most similar   
            query_len = len(query.split())
            simi_phrase_len = len(simi_phrase.split())
            if(query_len > 0 and simi_phrase_len > 0):
                len_ratio = min(query_len/simi_phrase_len , simi_phrase_len/query_len)
                if(similarity >= len_ratio * 100):    # similarity of 'book' and 'second companion book' is 32 < 1/3 * 100
                    return simi_phrase
 
    return None
        
def dedup_nodes_in_graph(G, grounded):
    def find_inclusion_duplicates(contains_dupes): 
        dup_sets = []
        for phrase in contains_dupes:
            rest_phrases = [p for p in contains_dupes if p != phrase] 
            inclusion_similar_phrase = inclusion_best_match(phrase, rest_phrases)
            if(inclusion_similar_phrase): 
                idx_phrase = [idx for idx, set in enumerate(dup_sets) if(phrase in set)]   # the set where phrase already in
                idx_inclusion_similar_phrase = [idx for idx, set in enumerate(dup_sets) if(inclusion_similar_phrase in set)]
                if(len(idx_phrase) > 0 and len(idx_inclusion_similar_phrase) == 0):
                    dup_sets[idx_phrase[0]].add(inclusion_similar_phrase)
                elif(len(idx_inclusion_similar_phrase) > 0 and len(idx_phrase) == 0):
                    dup_sets[idx_inclusion_similar_phrase[0]].add(phrase)
                elif(len(idx_inclusion_similar_phrase) > 0 and len(idx_phrase) > 0):
                    if(idx_phrase[0] != idx_inclusion_similar_phrase[0]):
                        dup_sets[idx_phrase[0]] = dup_sets[idx_phrase[0]] | dup_sets[idx_inclusion_similar_phrase[0]]
                        dup_sets.pop(idx_inclusion_similar_phrase[0])
                elif(len(idx_inclusion_similar_phrase) == 0 and len(idx_phrase) == 0):
                    dup_sets.append(set([phrase, inclusion_similar_phrase]))
                else:
                    print("len(idx_inclusion_similar_phrase), len(idx_phrase)")
                    print(len(idx_inclusion_similar_phrase), len(idx_phrase) )
 
        return dup_sets

    def merge_dup_nodes(G, dup_sets, grounded):

        for node_sets in dup_sets:     
            assert len(node_sets) >= 2
            # for each set, decide which one to be merged to
            # longest phrase that is same as a question phrase 
            merged_node = sorted([p for p in node_sets if p in grounded], key=lambda x: len(x), reverse=True)
            if(len(merged_node) == 0):   
                merged_node = sorted(node_sets, key=lambda x: len(x), reverse=True) # longest phrase
                        
            # merged_node[0] is the node to be merged to for current node_set
            for n in node_sets:
                if(n != merged_node[0]):
                    G = nx.contracted_nodes(G, merged_node[0], n)   # merge node
                        
        return G
    
    dup_sets = find_inclusion_duplicates(G.nodes)
    return merge_dup_nodes(G, dup_sets, grounded), dup_sets

def construct_reduced_context(raw_contexts, paras_phrases, extended_phrases, mapping):    
    raw_reduced_contexts = []     # sentences contain one of the extended_phrases
    number_sentences = 0
    number_reduced_sentences = 0 
    kept_para_sent = []
    for para_id, (para_title, para_lines) in enumerate(raw_contexts):  
        number_sentences += len(para_lines)
        reduced_para = []
        kept_sent = []
        for sent_id, sent in enumerate(para_lines):
            sentence_phrases = list(flatten(paras_phrases[para_id][sent_id+1]))[::2]  # paras_phrases[para_id][0] are phrases from the title, every other element is text, others are rank  
            if(any([sentence_phrase in mapping for sentence_phrase in sentence_phrases])): # at least one of sentence_phrase mapped to question phrase
                reduced_para.append(sent)
                number_reduced_sentences += 1 
                kept_sent.append(sent_id)
                continue

            for phrase in extended_phrases:                    
                if(phrase in sentence_phrases):  # current sentence has a exact match to extended_phrases 
                    reduced_para.append(sent)
                    number_reduced_sentences += 1 
                    kept_sent.append(sent_id)
                    break     # no need to continue checking whether current sentence contains other extended_phrases

        if(len(reduced_para) > 0):
            raw_reduced_contexts.append([para_title, reduced_para])
            kept_para_sent.append(kept_sent)
        else:
            for phrase in extended_phrases:
                if(phrase in list(flatten(paras_phrases[para_id][0]))[::2]):   # only tilte contains one of the extended_phrases
                    raw_reduced_contexts.append([para_title, []])
                    kept_para_sent.append(kept_sent)
                    break
                      
    assert number_reduced_sentences <= number_sentences    

    return raw_reduced_contexts, kept_para_sent

def construct_reduced_supporting_facts(supporting_facts, reduced_contexts, kept_para_sent):
    
    reduced_supporting_facts = []
    reduced_supporting_facts_in_original_id = []
    support_para = set(
        para_title for para_title, _ in supporting_facts
    )
    sp_set = set(list(map(tuple, supporting_facts)))                              # a list of (title, sent_id in orignal context) 

    for i, para_reduced_context in enumerate(reduced_contexts):                   # each para
        if(para_reduced_context[0] in support_para):
            for sent_id, orig_sent_id in enumerate(kept_para_sent[i]):
                if( (para_reduced_context[0], orig_sent_id) in sp_set ):
                    reduced_supporting_facts.append([para_reduced_context[0], sent_id])
                    reduced_supporting_facts_in_original_id.append([para_reduced_context[0], orig_sent_id])
                    
    return reduced_supporting_facts, reduced_supporting_facts_in_original_id


# revised for extractiing phrases, case matters for phrases extraction
def lower(text):
    return text.lower()

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
 
def basic_normalize(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def replace_special_punc(text):
        sentence_end = set(['.', '?', '!'])
        chs = []
        for ch in str(text):
            if ch in sentence_end:
                chs.append(',')
            elif ch == '-':
                chs.append(' ')
            else:
                chs.append(ch)
        
        return ''.join(chs)

#     def remove_stop_words(text):
#         all_stopwords = set(nlp.Defaults.stop_words)
#         return ' '.join(word for word in text.split() if word not in all_stopwords) 
    def remove_wh_words(text):
        wh_words = set(["what", "when", 'where', "which", "who", "whom", "whose", "why", "how", "whether",
                        "What", "When", 'Where', "Which", "Who", "Whom", "Whose", "Why", "How", "Whether"])
        return ' '.join(word for word in text.split() if word not in wh_words) 

    return white_space_fix(remove_wh_words(replace_special_punc(str(s))))

def _normalize_text(s):
    return basic_normalize(remove_articles(remove_punc(lower(str(s)))))
 
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
