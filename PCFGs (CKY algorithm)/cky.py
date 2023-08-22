"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer

Name:Xiaoxue Xiong
Uni:xx2385
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        n=len(tokens)
        table=dict()
        for i in range(0,n):
            table[(i,i+1)]=list()
            ls=self.grammar.rhs_to_rules.get((tokens[i],))
            for l in ls:
                table[(i,i+1)].append([l[0],l[1][0]])


        for length in range(2,n+1):
            for i in range(0,n-length+1):
                j=i+length
                table[(i,j)]=list()
                for k in range(i+1,j):
                    if table[(i,k)] and table[(k,j)] is not None:
                        for key in table[(i,k)]:
                            for item in table[(k,j)]:
                                rhs=(key[0],item[0])
                                if slef.grammar.rhs_to_rules.get(rhs) is not None:
                                    for l in self.grammar.rhs_to_rules.get(rhs):
                                        table[(i,j)].append([l[0],((l[1][0],i,k),(l[1][1],k,j))])

        if self.grammar.startsymbol in [i[0] for i in table[(0,6)]]:
            return True
        else:
            return False

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        n=len(tokens)
        table=dict()
        probs=dict()
        for i in range(0,n):
            table[(i,i+1)]=dict()
            probs[(i,i+1)]=dict()
            ls=self.grammar.rhs_to_rules.get((tokens[i],))
            if ls is not None:
                for l in ls:
                    table[(i,i+1)].update({l[0]:l[1][0]})
                    probs[(i,i+1)].update({l[0]:math.log(l[2])})
            else:
                return False,False


        for length in range(2,n+1):
            for i in range(0,n-length+1):
                j=i+length
                table[(i,j)]=dict()
                probs[(i,j)]=dict()
                for k in range(i+1,j):
                    if table[(i,k)] and table[(k,j)] is not None:
                        for key in table[(i,k)].keys():
                            for item in table[(k,j)].keys():
                                rhs=(key,item)
                                if self.grammar.rhs_to_rules.get(rhs) is not None:
                                    for l in self.grammar.rhs_to_rules.get(rhs):
                                        pro=math.log(l[2])+probs[(i,k)][l[1][0]]+probs[(k,j)][l[1][1]]
                                        if l[0] in list(table[(i,j)].keys()):
                                            if pro>=probs[(i,j)][l[0]]:
                                                probs[(i,j)].update({l[0]:pro})
                                                table[(i,j)].update({l[0]:((l[1][0],i,k),(l[1][1],k,j))})

                                        else:
                                            probs[(i,j)].update({l[0]:pro})
                                            table[(i,j)].update({l[0]:((l[1][0],i,k),(l[1][1],k,j))})
        
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    
    if j-i>1:
        value=chart[(i,j)][nt]
        sp1=get_tree(chart,value[0][1],value[0][2],value[0][0])
        sp2=get_tree(chart,value[1][1],value[1][2],value[1][0])
        return (nt,sp1,sp2)
    else:
        return (nt,chart[(i,j)][nt])
  
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        #print(parser.is_in_language(toks))
        #table,probs = parser.parse_with_backpointers(toks)
        #assert check_table_format(chart)
        #assert check_probs_format(probs)
        
