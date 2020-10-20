import numpy as np
import os
import random
from catboost import CatBoostClassifier, Pool

import joblib

import io
import transformers
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from zipfile import ZipFile
import torch

import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
from sklearn import preprocessing

def read_single_training_file(path):
    ## This method reads in a cupt file and writes it to a 2-dimensional array where the first dimension refers to sentences, and the second dimension to tokens within this sentence.
    ## The tokens themselves as well as their annotations are accessible by the resp. third dimension
    multiwords = {}
    head = None
    sentence_metadata = {}
    with open(path,'r',encoding='utf8') as f:
        sent = []
        sents = []
        sentence_metadata[0] = []
        metadata_mode = False
        for line in f:
            line_strip = line.strip()
            if len(line_strip) > 0 and line_strip[0] == '#':
                sentence_metadata[len(sents)].append(line_strip)
                metadata_mode = True
            elif len(line_strip) > 0:
                metadata_mode = False
                tk = line_strip.split('\t')
                if '-' in tk[0]:
                    if not len(sents) in multiwords:
                        multiwords[len(sents)] = {}
                    multiwords[len(sents)][int(tk[0].split('-')[0]) - 1] = line_strip[:-1] + '*'
                    continue

                sent.append(tk)
            else:
                if len(sent) > 0:
                    sents.append(sent)
                    sent = []
                    sentence_metadata[len(sents)] = []
        sents.append(sent)
        f.close()
        return sents, multiwords, sentence_metadata

def get_mwe_tags(sents):
    # This method extracts all observed MWE tags from a dataset
    tags = []
    tags.append('*')
    for sent in sents:
        for elem in sent:
            split_tags = elem[10].split(';')
            for tag in split_tags:
                split_tag = tag.split(':')
                if len(split_tag) > 1 and not split_tag[1] in tags:
                    tags.append(split_tag[1])              
    return tags

def get_token_lemma_pos_dep_tags(sents,po_tags,de_tags,toke,lemma):
    # This method extracts all observed POS- and dependency tags as well as lemmata observed for MWEs
    pos_tags = []
    if po_tags != None:
        pos_tags = po_tags
    dep_tags = []
    if de_tags != None:
        dep_tags = de_tags
    tokens = []
    if toke != None:
        tokens = toke
    lemmata = []
    if lemma != None:
        lemmata = lemma
    for sent in sents:
        for elem in sent:
            if not elem[4] in pos_tags:
                pos_tags.append(elem[4])
            if not elem[7] in dep_tags:
                dep_tags.append(elem[7])
            if not elem[2] in lemmata and elem[10] != '*':
                lemmata.append(elem[2])
    return pos_tags, dep_tags, tokens, lemmata

def get_morph_info(sents,morphtags):
    # This method extracts all observed morpho-semantic features from a dataset
    morph_tag_categories = []
    if morphtags != None:
        morph_tag_categories = morphtags
    for sent in sents:
        for elem in sent:
            split_cats = elem[5].split('|')
            for split_cat in split_cats:
                if not split_cat in morph_tag_categories:
                    morph_tag_categories.append(split_cat)
    return morph_tag_categories

def get_n_hot_vector(length,indices,start_index,vector):
    # This method is used for writing n-hot vectors to the respective part within the feature vector
    for i in range(length):
        if i in indices:
            vector[start_index + i] = 1.0
    return start_index + length

def encode_tag(elem,tag_index,tagset,start_index,vector):
    # This method is used for encoding tags as n-hot vectors
    si = start_index
    if elem[tag_index] in tagset:
        si = get_n_hot_vector(len(tagset),[tagset.index(elem[tag_index])],start_index,vector)
    else:
        si = get_n_hot_vector(len(tagset),[],start_index,vector)

    return si

def encode_children(elem,sent,tag_index,tagset,start_index,vector):
    # This method is used for encoding the children of a token within the dependency tree as n-hot vectors
    indices = set()
    for e in sent:
        if e[6] == elem[0] and e[tag_index] in tagset:
            indices.add(tagset.index(e[tag_index]))

    return get_n_hot_vector(len(tagset),indices,start_index,vector)

def encode_children_gram_morph(elem,sent,tagset,start_index,vector):
    # This method is used for encoding the gram-morph children of a token within the dependency tree as n-hot vectors
    indices = set()
    for e in sent:
        if e[6] == elem[0]:
            spl = e[5].split('|')
            for ff in spl:
                if ff in tagset:
                    indices.add(tagset.index(ff))

    return get_n_hot_vector(len(tagset),indices,start_index,vector)

def encode_siblings(elem,sent,tag_index,tagset,start_index,vector):
    # This method is used for encoding the siblings of a token within the dependency tree as n-hot vectors
    indices = set()
    for e in sent:
        if e[6] == elem[6] and e[tag_index] in tagset:
            indices.add(tagset.index(e[tag_index]))

    return get_n_hot_vector(len(tagset),indices,start_index,vector)

def encode_siblings_gram_morph(elem,sent,tagset,start_index,vector):
    # This method is used for encoding the gream morph information of the siblings of a token within the dependency tree as n-hot vectors
    indices = set()
    for e in sent:
        if e[6] == elem[6]:
            spl = e[5].split('|')
            for ff in spl:
                if ff in tagset:
                    indices.add(tagset.index(ff))

    return get_n_hot_vector(len(tagset),indices,start_index,vector)

def encode_gram_morph_tags(elem,gram_morph,start_index,vector):
    # This method is used for encoding the morpho-semantic information of a token
    split = elem[5].split('|')

    for spl in split:
        if spl in gram_morph:
            vector[start_index + gram_morph.index(spl)] = 1.0
    
    return start_index + len(gram_morph)

def encode_left_context(sent,elem,tag_index,tagset,context_size,start_index,vector):
    # Method for encoding a left context window
    start_index = start_index
    for i in range(int(elem[0]) - 1 - context_size,int(elem[0]) -1):
        if i >= 0 and elem[tag_index] in tagset:
            start_index = encode_tag(sent[i],tag_index,tagset,start_index,vector)
        else:
            start_index = get_n_hot_vector(len(tagset),[],start_index,vector)
    return start_index

def encode_right_context(sent,elem,tag_index,tagset,context_size,start_index,vector):
    # Method for encoding a right context window
    start_index = start_index
    for i in range(int(elem[0]) - 1, int(elem[0]) - 1 + context_size):
        if i < len(sent) and elem[tag_index] in tagset:
            start_index = encode_tag(sent[i],tag_index,tagset,start_index,vector)
        else:
            start_index = get_n_hot_vector(len(tagset),[],start_index,vector)
    return start_index

def encode_left_context_gram_morph(sent,i,gram_morph_tags,context_size,start_index,vector):
    kk = 0
    # Left context window for morphosemantics
    for c in range(context_size):
        if i - c <= 0:
            kk += len(gram_morph_tags)
            continue

        split = sent[i - c][5].split('|')

        for t in range(len(gram_morph_tags)):
            if gram_morph_tags[t] in split:
                vector[start_index + t] = 1

        kk += len(gram_morph_tags)

    return start_index + kk

def encode_right_context_gram_morph(sent,i,gram_morph_tags,context_size,start_index,vector):
    kk = 0
    # right context window for morphosemantics
    for c in range(context_size):
        if i  + c >= len(sent) - 1:
            kk += len(gram_morph_tags)
            continue

        split = sent[i + 1 + c][5].split('|')

        for t in range(len(gram_morph_tags)):
            if gram_morph_tags[t] in split:
                vector[start_index + t] = 1

        kk += len(gram_morph_tags)

    return start_index + kk
        
def encode_sentence_xlm_roberta(sent,model,tokenizer):
    # This method encodes roberta embeddings and attention values
    elems_len = {}
    #ids = []
    root_id = 0

    bad_sentence = False

    id_cons = {}
    ids_len = 0

    id_cons[0] = [0]
    for i in range(len(sent)):
        elem = sent[i]
        enc = tokenizer.encode(elem[1])

        if elem[6] == '0':
            root_id = i
    
        if len(enc) > 3:
            elems_len[i] = len(enc) - 2
            id_cons[i + 1] = [enc[k] for k in range(1, len(enc) - 1)]
            ids_len += len(id_cons[i + 1])
        else:
            elems_len[i] = 1
            id_cons[i + 1] = [enc[1]]
            ids_len += 1

    id_cons[len(sent)] = [2]
    
    ids = [0] * (ids_len + 2)
    id_idx = 0
    for k in range(len(id_cons)):
        for jj in id_cons[k]:
            ids[id_idx] = jj
            id_idx += 1


    if len(ids) > 512:
        ids = ids[:512]

    as_tenser = torch.LongTensor(ids)[None,:]

    out = model(as_tenser)
    #print(out)
    logits = torch.softmax(out[2][0][0],dim=1)
    #logits = out[2][0][0]
    attention_weights = out[3][1][0]

    final_reps_logits = [None] * len(sent)
    final_reps_attention_tp = [None] * len(sent)
    final_reps_attention_fp = [None] * len(sent)
    final_reps_attention_tr = [None] * len(sent)
    final_reps_attention_fr = [None] * len(sent)

    nav_i = 0
    for i in range(len(sent)):
        elem = sent[i]
        parent_id = elem[6]
        if (int(parent_id) - 1) >= len(elems_len):
            parent_id = str(len(elems_len))
        if (int(parent_id) - 1) < 0:
            parent_id = '0'
        #ugly hotfix
        if i >= 512:
            final_reps_attention_tp[i] = final_reps_attention_tp[512]
            final_reps_attention_fp[i] = final_reps_attention_fp[512]

            final_reps_attention_tr[i] = final_reps_attention_tr[512]
            final_reps_attention_fr[i] = final_reps_attention_tr[512]

            final_reps_logits[i] = final_reps_logits[len(ids) - 1]
            continue

        if elems_len[i] > 1:
            temp = set()
            temp_att_tp_e = set()
            temp_att_fp_e = set()
            temp_att_tr_e = set()
            temp_att_fr_e = set()
            for k in range(elems_len[i]):
                temp.add(logits[i + k + 1])
                temp_att_tp = [0] * len(attention_weights)
                temp_att_fp = [0] * len(attention_weights)
                temp_att_tr = [0] * len(attention_weights)
                temp_att_fr = [0] * len(attention_weights)
                for t in range(len(attention_weights)):
                    if parent_id == '0':
                        if elems_len[i] > 1:
                            temp_tp = [None] * elems_len[i]
                            temp_fp = [None] * elems_len[i]
                            for h in range(elems_len[i]):
                                temp_tp[h] = attention_weights[t][i + k + 1][i + 1 + h]
                                temp_fp[h] = attention_weights[t][i + 1 + h][i + k + 1]

                            temp_att_tp[t] = torch.mean(torch.stack(temp_tp))                           
                            temp_att_fp[t] = torch.mean(torch.stack(temp_fp))
                        else:
                            temp_att_tp[t] = attention_weights[t][i + k + 1][i + k + 1]
                            temp_att_fp[t] = attention_weights[t][i + k + 1][i + k + 1]

                        if elems_len[root_id] > 1:
                            temp_tr = [None] * elems_len[root_id]
                            temp_fr = [None] * elems_len[root_id]
                            for h in range(elems_len[root_id]):
                                temp_fr[h] = attention_weights[t][root_id + 1 + h][i + k + 1]
                                temp_tr[h] = attention_weights[t][i + k + 1][root_id + 1 + h]

                            temp_att_tr[t] = torch.mean(torch.stack(temp_tr))
                            temp_att_fr[t] = torch.mean(torch.stack(temp_fr))
                        else:
                            temp_att_fr[t] = attention_weights[t][root_id + 1][i + k + 1]
                            temp_att_tr[t] = attention_weights[t][i + k + 1][root_id + 1]

                    else:
                        if elems_len[int(parent_id) - 1] > 1:
                            temp_tp = [None] * elems_len[int(parent_id) - 1]
                            temp_fp = [None] * elems_len[int(parent_id) - 1]
                            for h in range(elems_len[int(parent_id) - 1]):
                                temp_tp[h] = attention_weights[t][i + k + 1][int(parent_id) + h]
                                temp_fp[h] = attention_weights[t][int(parent_id) + h][i + k + 1]

                            temp_att_tp[t] = torch.mean(torch.stack(temp_tp))                          
                            temp_att_fp[t] = torch.mean(torch.stack(temp_fp))
                        else:
                            temp_att_tp[t] = attention_weights[t][i + k + 1][int(parent_id)]
                            temp_att_fp[t] = attention_weights[t][int(parent_id)][i + k + 1]

                        if elems_len[root_id] > 1:
                            temp_tr = [None] * elems_len[root_id]
                            temp_fr = [None] * elems_len[root_id]
                            for h in range(elems_len[root_id]):
                                temp_fr[h] = attention_weights[t][root_id + 1 + h][i + k + 1]
                                temp_tr[h] = attention_weights[t][i + k + 1][root_id + 1 + h]

                            temp_att_tr[t] = torch.mean(torch.stack(temp_tr))
                            temp_att_fr[t] = torch.mean(torch.stack(temp_fr))
                        else:
                            temp_att_fr[t] = attention_weights[t][root_id + 1][i + k + 1]
                            temp_att_tr[t] = attention_weights[t][i + k + 1][root_id + 1]

                temp_att_tp_e.add(torch.stack(temp_att_tp))
                temp_att_fp_e.add(torch.stack(temp_att_fp))
                temp_att_fr_e.add(torch.stack(temp_att_fr))
                temp_att_tr_e.add(torch.stack(temp_att_tr))

            final_reps_attention_tp[i] = torch.mean(torch.stack(list(temp_att_tp_e)),dim=0)
            final_reps_attention_fp[i] = torch.mean(torch.stack(list(temp_att_fp_e)),dim=0)
            final_reps_attention_tr[i] = torch.mean(torch.stack(list(temp_att_tr_e)),dim=0)
            final_reps_attention_fr[i] = torch.mean(torch.stack(list(temp_att_fr_e)),dim=0)
            
            final_reps_logits[i] = torch.mean(torch.stack(list(temp)),dim=0)

        else:
            temp_att_tp_e = [0] * len(attention_weights)
            temp_att_fp_e = [0] * len(attention_weights)
            temp_att_tr_e = [0] * len(attention_weights)
            temp_att_fr_e = [0] * len(attention_weights)
            for t in range(len(attention_weights)):
                if parent_id == '0':
                    temp_att_tp_e[t] = attention_weights[t][i + 1][i + 1]
                    temp_att_fp_e[t] = attention_weights[t][i + 1][i + 1]

                    if elems_len[root_id] > 1:
                        temp_tr = [None] * elems_len[root_id]
                        temp_fr = [None] * elems_len[root_id]
                        for h in range(elems_len[root_id]):
                            temp_fr[h] = attention_weights[t][root_id + 1 + h][i + 1]
                            temp_tr[h] = attention_weights[t][i + 1][root_id + 1 + h]

                        temp_att_tr_e[t] = torch.mean(torch.stack(temp_tr))
                        temp_att_fr_e[t] = torch.mean(torch.stack(temp_fr))
                    else:
                        temp_att_fr_e[t] = attention_weights[t][root_id + 1][i + 1]
                        temp_att_tr_e[t] = attention_weights[t][i + 1][root_id + 1]

                else:
                    if elems_len[int(parent_id) - 1] > 1:
                        temp_tp = [None] * elems_len[int(parent_id) - 1]
                        temp_fp = [None] * elems_len[int(parent_id) - 1]
                        for h in range(elems_len[int(parent_id) - 1]):
                            temp_tp[h] = attention_weights[t][i + 1][int(parent_id) + h]
                            temp_fp[h] = attention_weights[t][int(parent_id) + h][i + 1]

                        temp_att_tp_e[t] = torch.mean(torch.stack(temp_tp))
                        temp_att_fp_e[t] = torch.mean(torch.stack(temp_fp))
                    else:
                        temp_att_tp_e[t] = attention_weights[t][i + 1][int(parent_id)]
                        temp_att_fp_e[t] = attention_weights[t][int(parent_id)][i + 1]

                    if elems_len[root_id] > 1:
                        temp_tr = [None] * elems_len[root_id]
                        temp_fr = [None] * elems_len[root_id]
                        for h in range(elems_len[root_id]):
                            temp_fr[h] = attention_weights[t][root_id + 1 + h][i + 1]
                            temp_tr[h] = attention_weights[t][i + 1][root_id + 1 + h]

                        temp_att_tr_e[t] = torch.mean(torch.stack(temp_tr))
                        temp_att_fr_e[t] = torch.mean(torch.stack(temp_fr))
                    else:
                        temp_att_fr_e[t] = attention_weights[t][root_id + 1][i + 1]
                        temp_att_tr_e[t] = attention_weights[t][i + 1][root_id + 1]

            final_reps_attention_tp[i] = torch.stack(temp_att_tp_e)
            final_reps_attention_fp[i] = torch.stack(temp_att_fp_e)

            final_reps_attention_tr[i] = torch.stack(temp_att_tr_e)
            final_reps_attention_fr[i] = torch.stack(temp_att_fr_e)

            final_reps_logits[i] = logits[i + 1]  

    return final_reps_logits, final_reps_attention_fp, final_reps_attention_tp, final_reps_attention_tr, final_reps_attention_fr, bad_sentence   

def encode_roberta_logits(logits,i,start_index,vector):
    for j in range(len(logits[i])):
        vector[start_index + j]  = logits[i][j]
    return start_index + len(logits[i])

def encode_attention_labs(attention_weights,i,start_index,vector):
    for k in range(len(attention_weights[i])):
        vector[start_index + k] = attention_weights[i][k]
    return start_index + len(attention_weights[i])

def read_dice_directory(directory_path,limit=-1):
    files = [f.path for f in os.scandir(directory_path)]
    le = len(files)
    if limit > -1:
        le = limit
    pre_ret = [None] * le
    i = 0
    for filepath in files:
        if i == limit:
            break
        print(str(i) + '/' + str(len(files)))
        pre_ret[i] = read_single_training_file(filepath)
        i += 1
    return [sent for file in pre_ret for sent in file]

def encode_children_roberta(sent,i,final_reps_logits,start_index,vector):
    st = set()
    for j in range(len(sent)):
        if sent[i][0] == sent[j][6] and i != j:
            st.add(final_reps_logits[j])

    if len(st) > 0:
        r = torch.mean(torch.stack(list(st)),dim=0)
        for k in range(len(r)):
            vector[start_index + k] = r[k]
    
    return start_index + len(final_reps_logits[i])

def encode_siblings_roberta(sent,i,final_reps_logits,start_index,vector):
    st = set()
    for j in range(len(sent)):
        if sent[i][6] == sent[j][6] and i != j:
            st.add(final_reps_logits[j])

    if len(st) > 0:
        r = torch.mean(torch.stack(list(st)),dim=0)
        for k in range(len(r)):
            vector[start_index + k] = r[k]

    return start_index + len(final_reps_logits[i])

def encode_sentence(
    sent,
    xlm_model,
    xlm_tokenizer,
    pos_tags,
    dep_tags,
    tokens,
    lemmata,
    gram_morph_tags,
    vector_pos,
    data_vector
):
    #final_reps_logits, final_reps_attention_fp, final_reps_attention_tp, final_reps_attention_tr, final_reps_attention_fr, bad_sentence = encode_sentence_xlm_roberta(sent,xlm_model,xlm_tokenizer)

    root_id = 0
    for i in range(len(sent)):
        if sent[i][7] == 'root':
            root_id = i
            break

    labels = [None] * len(sent)
    digit_heads = {}
    for i in range(len(sent)):
        elem = sent[i]
        start_index = 0
        vector = data_vector[vector_pos + i]
        
        #start_index = encode_roberta_logits(final_reps_logits,i,start_index,vector)
        #start_index = encode_roberta_logits(final_reps_logits,root_id,start_index,vector)
        
        #if root_id == i:
        #    start_index = encode_roberta_logits(final_reps_logits,i,start_index,vector)
        #else:
        #    start_index = encode_roberta_logits(final_reps_logits,int(elem[6]) - 1,start_index,vector)

        #start_index = encode_children_roberta(sent,i,final_reps_logits,start_index,vector)
        #start_index = encode_siblings_roberta(sent,i,final_reps_logits,start_index,vector)

        #start_index = encode_attention_labs(final_reps_attention_fp,i,start_index,vector)
        #start_index = encode_attention_labs(final_reps_attention_tp,i,start_index,vector)
        #start_index = encode_attention_labs(final_reps_attention_tr,i,start_index,vector)
        #start_index = encode_attention_labs(final_reps_attention_fr,i,start_index,vector)

        start_index = encode_children(elem,sent,2,lemmata,start_index,vector)
        start_index = encode_children(elem,sent,4,pos_tags,start_index,vector)
        start_index = encode_children(elem,sent,7,dep_tags,start_index,vector)

        start_index = encode_siblings(elem,sent,2,lemmata,start_index,vector)
        start_index = encode_siblings(elem,sent,4,pos_tags,start_index,vector)
        start_index = encode_siblings(elem,sent,7,dep_tags,start_index,vector)

        start_index = encode_tag(elem,2,lemmata,start_index,vector)
        
        start_index = encode_tag(elem,4,pos_tags,start_index,vector)
        start_index = encode_left_context(sent,elem,4,pos_tags,2,start_index,vector)
        start_index = encode_right_context(sent,elem,4,pos_tags,2,start_index,vector)

        start_index = encode_tag(elem,7,dep_tags,start_index,vector)
        start_index = encode_left_context(sent,elem,7,dep_tags,2,start_index,vector)
        start_index = encode_right_context(sent,elem,7,dep_tags,2,start_index,vector)

        if i != root_id:
            start_index = encode_tag(sent[int(elem[6]) - 1],2,lemmata,start_index,vector)
        else:
            start_index += len(lemmata)
        
        if i != root_id:
            start_index = encode_tag(sent[int(elem[6]) - 1],4,pos_tags,start_index,vector)
            start_index = encode_right_context(sent,sent[int(elem[6]) - 1],4,pos_tags,2,start_index,vector)
            start_index = encode_left_context(sent,sent[int(elem[6]) - 1],4,pos_tags,2,start_index,vector)
        else:
            start_index += len(pos_tags) * 5

        if i != root_id:
            start_index = encode_tag(sent[int(elem[6]) - 1],7,dep_tags,start_index,vector)
            start_index = encode_right_context(sent,sent[int(elem[6]) - 1],7,dep_tags,2,start_index,vector)
            start_index = encode_left_context(sent,sent[int(elem[6]) - 1],7,dep_tags,2,start_index,vector)
        else:
            start_index += len(dep_tags) * 5

        start_index = encode_tag(sent[root_id],4,pos_tags,start_index,vector)
        start_index = encode_right_context(sent,sent[root_id],4,pos_tags,2,start_index,vector)
        start_index = encode_left_context(sent,sent[root_id],4,pos_tags,2,start_index,vector)

        start_index = encode_tag(sent[root_id],7,dep_tags,start_index,vector)
        start_index = encode_right_context(sent,sent[root_id],7,dep_tags,2,start_index,vector)
        start_index = encode_left_context(sent,sent[root_id],7,dep_tags,2,start_index,vector)

        start_index = encode_tag(sent[root_id],2,lemmata,start_index,vector)

        start_index = encode_gram_morph_tags(elem,gram_morph_tags,start_index,vector)

        start_index = encode_left_context_gram_morph(sent,i,gram_morph_tags,2,start_index,vector)
        start_index = encode_right_context_gram_morph(sent,i,gram_morph_tags,2,start_index,vector)

        if i != root_id:         
            start_index = encode_gram_morph_tags(sent[int(elem[6]) - 1],gram_morph_tags,start_index,vector)
        else:
            start_index += len(gram_morph_tags)

        start_index = encode_gram_morph_tags(sent[root_id],gram_morph_tags,start_index,vector)

        start_index = encode_children_gram_morph(elem,sent,gram_morph_tags,start_index,vector)
        start_index = encode_siblings_gram_morph(elem,sent,gram_morph_tags,start_index,vector)

        split_tags = elem[10].split(';')
        for tag in split_tags:
            split_tag = tag.split(':')
            if len(split_tag) > 1:
                if not split_tag[0] in digit_heads:
                    digit_heads[split_tag[0]] = split_tag[1]

    for i in range(len(sent)):
        elem = sent[i]
        split_tags = elem[10].split(';')
        labs = set()
        for tag in split_tags:
            split_tag = tag.split(':')
            if len(split_tag) > 1:
                labs.add(split_tag[1])
            elif split_tag[0].isdigit():
                labs.add(digit_heads[split_tag[0]])
        if len(labs) == 0:
            labs.add('*')
                
        labels[i] = labs

    return labels, vector_pos + len(sent)

def encode_dice_score(dice_scores,first_token,second_token,vector,start_index):
    if first_token in dice_scores and second_token in dice_scores[first_token]:
        vector[start_index] = dice_scores[first_token][second_token]

    return start_index + 1

def encode_sentences(in_set,
    xlm_model,
    xlm_tokenizer,
    pos_tags,
    dep_tags,
    tokens,
    lemmata,
    gram_morph_tags
):
    print('allocating vector memory')
    len_enc_vec = 5 * xlm_model.config.hidden_size 
    len_enc_vec += 4 * xlm_model.config.num_attention_heads 
    len_enc_vec += 5 * len(lemmata)
    len_enc_vec += 17 * len(pos_tags)
    len_enc_vec += 17 * len(dep_tags)
    len_enc_vec += 9 * len(gram_morph_tags)
    num_tks = 0
    for s in in_set:
        for _ in s:
            num_tks += 1
    inputs = np.zeros(shape=(num_tks,len_enc_vec),dtype=np.float64)
    labels_temp = {}
    labels_len = 0
    i = 0
    vector_pos = 0
    print('iterating')
    for i in range(len(in_set)):
        if i % 20 == 0:
            print(str(i) + '/' + str(len(in_set)))
        labs, vector_pos = encode_sentence(
            in_set[i],
            xlm_model,
            xlm_tokenizer,
            pos_tags,
            dep_tags,
            tokens,
            lemmata,
            gram_morph_tags,
            vector_pos,
            inputs
        )

        labels_temp[i] = labs
        labels_len += len(labs)

    labels = [None] * labels_len

    current_idx = 0
    for k in range(len(in_set)):
        for j in labels_temp[k]:
            labels[current_idx] = j
            current_idx += 1

    return inputs, labels

def reconstruct_cupt_label_format(sents,predicted_labels,mwe_tags):
    ret_for_sents = [[] for x in range(len(sents))]
    idx_last_seen = 0

    labels_counter = 0

    def gt(a):
        return a[0]

    for i in range(len(sents)):
        sent = sents[i] 

        ret_for_sents[i] = [['*'] for x in range(len(sent))]

        idxs = {}

        for tag in mwe_tags:
            if tag == '*':
                continue

            labels_counter_int = labels_counter

            graph = nx.Graph()

            for t in range(len(sent)):
                if predicted_labels[tag][labels_counter_int] == 1:
                    graph.add_node(t)

                labels_counter_int += 1

            nds = list(graph.nodes)

            for t in range(len(sent)):
                if t in nds and (int(sent[t][6]) - 1) in nds:
                    graph.add_edge((int(sent[t][6]) - 1),t)

            connected_components = list(nx.connected_components(graph))
            conn = []
            for c in range(len(connected_components)):
                if len(connected_components[c]) > 0:
                    conn.append(sorted(list(connected_components[c])))

            conn.sort(key=gt)

            for s in conn:
                if s[0] in idxs:
                    idxs[s[0]].append((tag,s))
                else:
                    idxs[s[0]] = []
                    idxs[s[0]].append((tag,s))

        idx_cnt = 1

        for k in sorted(list(idxs.keys())):
            for elem in idxs[k]:
                if '*' in ret_for_sents[i][elem[1][0]]:
                    ret_for_sents[i][elem[1][0]].remove('*')
                ret_for_sents[i][elem[1][0]].append((str(idx_cnt) + ':' + elem[0]))
                ret_for_sents[i][elem[1][0]].sort()
                for f in elem[1][1:]:
                    if '*' in ret_for_sents[i][f]:
                        ret_for_sents[i][f].remove('*')
                    ret_for_sents[i][f].append(str(idx_cnt))
                    ret_for_sents[i][f].sort()
                
                idx_cnt += 1
            
        labels_counter += len(sent)

    return ret_for_sents

def reconstruct_cupt_string(sents,multiwords,sentence_metadata,ret_for_sents):
    ret_str = ''
    for i in range(len(sents)):

        for metadata in sentence_metadata[i]:
            ret_str += metadata + '\n'

        for j in range(len(sents[i])):

            if i in multiwords and j in multiwords[i]:
                ret_str += multiwords[i][j] + '\n'

            line_c = ''
            for k in range(10):
                line_c += sents[i][j][k] + '\t'
            
            if len(ret_for_sents[i][j]) > 1:
                for e in ret_for_sents[i][j]:
                    line_c += e + ';'
                line_c = line_c[:-1]
            else:
                line_c += ret_for_sents[i][j][0]
            ret_str += line_c + '\n'
        ret_str += '\n'
    return ret_str

def transform_labels_for_class(class_tag,labels):
    ret = []
    for i in range(len(labels)):
        if class_tag in labels[i]:
            ret.append(1)
        else:
            ret.append(0)
    return ret

def correct_dataset(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if dataset[i,j] == None:
                dataset[i,j] = 0.0

print('unzipping data')
with ZipFile('./data.zip','r') as zi:
    zi.extractall()

print('loading XLM RoBERTa')
xlm_roberta_config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
xlm_roberta_config.output_attentions = True
xlm_roberta_config.output_hidden_states = True
xlm_model = XLMRobertaModel.from_pretrained('xlm-roberta-base',config=xlm_roberta_config)
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# ['HI','GA','SV','EU','DE','EL','FR','HE','IT']
#['DE','EL','EU','FR','GA','HE','HI','IT','PL','PT','RO','SV','TR','ZH']
for lang in ['TR']:
    print(lang)

    print('loading train set')
    train_sents, train_multiwords, train_metadata = read_single_training_file('./data/' + lang + '/train.cupt')
    print('loading dev set')
    dev_sents, dev_multiwords, dev_metadata = read_single_training_file('./data/' + lang + '/dev.cupt')
    print('loading val set')
    validate_sents, validate_multiwords, validate_metadata = read_single_training_file('./data/' + lang + '/test.cupt')
    train_sents.extend(dev_sents)

    print('collecting tagsets')
    mwe_tags = get_mwe_tags(train_sents)
    pos_tags, dep_tags, tokens, lemmata = get_token_lemma_pos_dep_tags(train_sents,None,None,None,None)
    gram_morph_tags = get_morph_info(train_sents,None)

    print('encoding set')

    inp, labels = encode_sentences(
        train_sents,
        xlm_model,
        xlm_tokenizer,
        pos_tags,
        dep_tags,
        tokens,
        lemmata,
        gram_morph_tags
    )

    inp_test, labels_test = encode_sentences(
        validate_sents,
        xlm_model,
        xlm_tokenizer,
        pos_tags,
        dep_tags,
        tokens,
        lemmata,
        gram_morph_tags
    )

    print('Training')

    model = {}
    model['mwe_tags'] = mwe_tags
    model['pos_tags'] = pos_tags
    model['dep_tags'] = dep_tags
    model['gram_morph_tags'] = gram_morph_tags
    model['xlm_model_name'] = 'xlm-roberta-base'
    f1_scores_lr = {}
    f1_scores_gb = {}

    labels_for_mwe_class_lr = {}
    labels_for_mwe_class_gb = {}

    for i in range(1,len(mwe_tags)):
        model[mwe_tags[i]] = {}
        print(mwe_tags[i])
        labels_train = np.array(transform_labels_for_class(mwe_tags[i],labels))
        labels_val = np.array(transform_labels_for_class(mwe_tags[i],labels_test))
        #joblib.dump(labels_train,'./' + lang + '_train_set_labels' + mwe_tags[i] + '.joblib.bz',compress=('bz2', 3))

        #classifier_lr = LogisticRegression(class_weight='balanced',max_iter=1000).fit(inp,labels_train)
        #model[mwe_tags[i]]['classifier_lr'] = classifier_lr
        #pred_lr = classifier_lr.predict(inp_test)
        #model[mwe_tags[i]]['val_pred_lr'] = pred_lr

        #labels_for_mwe_class_lr[mwe_tags[i]] = pred_lr

        #f1_scores_lr[mwe_tags[i]] = f1_score(labels_val,pred_lr,average=None)        
        #print(f1_scores_lr[mwe_tags[i]])
        #print(precision_score(labels_val,pred_lr,average=None))
        #print(recall_score(labels_val,pred_lr,average=None))

        input_pool = Pool(inp,label=labels_train)
        eval_pool = Pool(inp_test)

        classifier_gb = CatBoostClassifier(verbose=True,task_type='CPU')
        classifier_gb.fit(input_pool,verbose=True)
        pred_gb = classifier_gb.predict(eval_pool)
        labels_for_mwe_class_gb[mwe_tags[i]] = pred_gb

        f1_scores_gb[mwe_tags[i]] = f1_score(labels_val,pred_gb,average=None)
        print(f1_scores_gb[mwe_tags[i]])
        print(precision_score(labels_val,pred_gb,average=None))
        print(recall_score(labels_val,pred_gb,average=None))

        model[mwe_tags[i]]['classifier_gb'] = './' + lang + '_catboost_' + mwe_tags[i] + '.cbm'
        model[mwe_tags[i]]['val_pred_gb'] = pred_gb
        classifier_gb.save_model('./' + lang + '_catboost_' + mwe_tags[i] + '.cbm',format='cbm')

    #ret_for_sents_val_lr = reconstruct_cupt_label_format(validate_sents,labels_for_mwe_class_lr,mwe_tags)
    #cupt_string_val_lr = reconstruct_cupt_string(validate_sents,validate_multiwords,validate_metadata,ret_for_sents_val_lr)

    ret_for_sents_val_gb = reconstruct_cupt_label_format(validate_sents,labels_for_mwe_class_gb,mwe_tags)
    cupt_string_val_gb = reconstruct_cupt_string(validate_sents,validate_multiwords,validate_metadata,ret_for_sents_val_gb)

    #with open('./data/' + lang +'/system.lr.cupt','w',encoding='utf8') as f:
    #    f.write(cupt_string_val_lr)
    #    f.close()

    with open('./data/' + lang +'/system.gb.cupt','w',encoding='utf8') as f:
        f.write(cupt_string_val_gb)
        f.close()

    joblib.dump(model, './' + lang + '_model.joblib.bz',compress=('bz2', 3))

    #with open('./' + lang + '_f1_lr.txt','w',encoding='utf8') as f:
    #    f.write(str(f1_scores_lr))
    #    f.close()

    with open('./' + lang + '_f1_gb.txt','w',encoding='utf8') as f:
        f.write(str(f1_scores_gb))
        f.close()