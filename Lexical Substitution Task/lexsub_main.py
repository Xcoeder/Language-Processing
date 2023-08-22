#!/usr/bin/env python
import sys
import string
from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    ls=wn.lemmas(lemma,pos)
    result=[]
    for l in ls:
        for m in l.synset().lemmas():
            if m.name()!=lemma:
                result.append(str(m.name()))
    return list(set(result))
 

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


class my_refinements(object):
    
    #combine the similarity and bert
    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.model2 = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict(self, context : Context) -> str:
        lemma=context.lemma
        pos=context.pos
        left_context=context.left_context
        right_context=context.right_context
        #obtain synonyms
        synonyms=get_candidates(lemma,pos)
        
        #step2
        left_context.append('[MASK]')
        left_context.extend(right_context)
        s="".join([x if x in string.punctuation else " "+x for x in left_context])
        s=s[1:]
        input_toks = self.tokenizer.encode(s)
        input_mat = np.array(input_toks).reshape((1,-1))
        mask_id=self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        
        
        #step3
        outputs = self.model.predict(input_mat)
        
        #step4
        predictions=outputs[0]
        best_words = np.argsort(predictions[0][mask_id])[::-1]
        words=self.tokenizer.convert_ids_to_tokens(best_words[:20])
        
        
        words.extend(synonyms)
        
        result=dict()
        for w in list(set(words)):
            if (w not in self.model.key_to_index) or (lemma not in self.model.key_to_index):
                continue
            else:
                result[w]=self.model.similarity(lemma,w)
            
        if len(result.keys())!=0:
            result=[k for k,v in result.items() if v == max(result.values())][0]
            return result 
        else:
            return words[0] 
        


def wn_frequency_predictor(context : Context) -> str:
    lem=context.lemma
    pos=context.pos
    
    ls=wn.lemmas(lem,pos)
    result=dict()
    for l in ls:
        for m in l.synset().lemmas():
            if m.name()!=lem:
                if m.name() not in result.keys():
                    result[m.name()]=m.count()
                else:
                    result[m.name()]+=m.count()
                
    
    return [k for k,v in result.items() if v == max(result.values())][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    lemma=context.lemma
    pos=context.pos
    ls=wn.lemmas(lemma,pos)
    left_context = context.left_context
    right_context =context.right_context
    left_context.extend(right_context)
    stop_words = stopwords.words('english')
    left_context=[i for i in left_context if not i in stop_words]
    overlap=dict()
    syn_freq=dict()
    for l in ls:
        syn_freq[l.synset()]=l.count()
        # construct definition
        s=l.synset()
        defi=s.definition()
        ex=s.examples()
        for e in ex:
            defi=defi+" "+e
        hyper=s.hypernyms()
        for h in hyper:
            defi=defi+" "+h.definition()
            for e in h.examples():
                defi=defi+" "+e
        # compute overlap
        defi=tokenize(defi)
        defi=[i for i in defi if not i in stop_words]
        count=0
        for d in defi:
            if d in left_context:
                count+=1
        overlap[s]=count
        
    #choose synset
    re=[k for k,v in overlap.items() if v == max(overlap.values()) and v!=0]
    if len(re)==0 or len(re)!=1:
        syn=[k for k,v in syn_freq.items()]
    else:
        syn=re
    
    #choose lexeme
    lexme=dict()
    for s in syn:
        for l in s.lemmas():
            lexme[l]=l.count()
            
    result=[k for k,v in lexme.items() if k.name()!=lemma]
    result=[k for k,v in lexme.items() if v == max(lexme.values())]
    result=result[0].name()
    
    return result
       
         

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        lemma=context.lemma
        pos=context.pos
        synonyms=get_candidates(lemma, pos)
        result=dict()
        for w in synonyms:
            if (w not in self.model.key_to_index) or (lemma not in self.model.key_to_index):
                continue
            else:
                result[w]=self.model.similarity(lemma,w)
            
        
        if len(result.keys())!=0:
            result=[k for k,v in result.items() if v == max(result.values())][0]
            return result 
        else:
            return None
             


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        lemma=context.lemma
        pos=context.pos
        left_context=context.left_context
        right_context=context.right_context
        #obtain synonyms
        synonyms=get_candidates(lemma,pos)
        
        #step2
        left_context.append('[MASK]')
        left_context.extend(right_context)
        s="".join([x if x in string.punctuation else " "+x for x in left_context])
        s=s[1:]
        input_toks = self.tokenizer.encode(s)
        input_mat = np.array(input_toks).reshape((1,-1))
        mask_id=self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        
        
        #step3
        outputs = self.model.predict(input_mat)
        
        #step4
        predictions=outputs[0]
        best_words = np.argsort(predictions[0][mask_id])[::-1]
        words=self.tokenizer.convert_ids_to_tokens(best_words)
        
        result=dict()
        for w in synonyms:
            if w in words:
                result[w]=words.index(w)
 
        if len(result.keys())!=0:
            best=[k for k,v in result.items() if v == min(result.values())][0]
        else:
            best=words[0]
        return best 

      

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #best=BertPredictor()
        #prediction =wn_frequency_predictor(context) 
        #prediction =wn_simple_lesk_predictor(context)
        
#         W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
#         best= Word2VecSubst(W2VMODEL_FILENAME)
#         prediction =best.predict_nearest(context)
        
        best=BertPredictor()
        prediction =best.predict(context)
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
