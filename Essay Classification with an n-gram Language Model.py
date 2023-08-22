

import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""
"""
Xiaoxue Xiong
xx2385
"""
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    se=['START']
    if n==1:
        se.extend(sequence)
        st=se
    else:
        st=sequence
        for i in range(0,n-1):
            se.extend(st)
            st=se
            se=['START']
    st.append('STOP')
    result=[]
    for i in range(0,len(st)-n+1):
        result.append(tuple(st[i:i+n]))
    return result


# In[141]:


class TrigramModel(object):
    
    def __init__(self, corpusfile):
        
#         corpusfile="./hw1_data/hw1_data/brown_train.txt"
        
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        #corpusfile="./hw1_data/hw1_data/brown_test.txt"
        
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        
        self.total_number=sum(self.unigramcounts.values())


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)
        
        for sentence in corpus:
            for unigram in get_ngrams(sentence,1):
                self.unigramcounts[unigram]+=1
            for bigram in get_ngrams(sentence,2):
                self.bigramcounts[bigram]+=1
            for trigram in get_ngrams(sentence,3):
                self.trigramcounts[trigram]+=1
            
        

        ##Your code here

        return  self.unigramcounts, self.bigramcounts, self.trigramcounts 

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        
        if self.bigramcounts[tuple(trigram[0:2])]!=0:
            result=self.trigramcounts[trigram]/self.bigramcounts[tuple(trigram[0:2])]
            return result
        else:
            return self.raw_unigram_probability(tuple(trigram[2:]))

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram=[]
        unigram.append(bigram[0])
        if self.unigramcounts[tuple(unigram)]!=0:
            result=self.bigramcounts[bigram]/self.unigramcounts[tuple(unigram)]        
            return result
        else:
            return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        result=self.unigramcounts[unigram]/self.total_number
        return result

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        unigram=[]
        unigram.append(trigram[2])        
        result=lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:3])+lambda3*self.raw_unigram_probability(tuple(unigram))
        return result
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams=get_ngrams(sentence,3)
        result=0.0
        for trigram in trigrams:
            prob=self.smoothed_trigram_probability(trigram)
            logprob=math.log2(prob)
            result+=logprob
            
        return result

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_number=0.0
        prob=0.0
        for sentence in corpus:
            for unigram in get_ngrams(sentence,1):
                total_number+=1
            total_number-=1
            prob+=self.sentence_logprob(sentence)
        result=math.pow(2,-prob/total_number)
        return result



# In[138]:


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1<=pp2:
                correct+=1
                total+=1
            else:
                total+=1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp<= model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon)):
                correct+=1
                total+=1
            else:
                total+=1
        
        return float(correct)/float(total)*100

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)


# In[ ]:




