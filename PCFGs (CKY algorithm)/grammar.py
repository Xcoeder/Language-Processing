"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer

Name:Xiaoxue Xiong
UNI:xx2385
"""

import sys
from collections import defaultdict
from math import fsum
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        nonterminals=list(self.lhs_to_rules.keys())
        for key in self.lhs_to_rules.keys():
            tmp_ls=self.lhs_to_rules.get(key)
            sum_pro=0
            A=True
            for item in tmp_ls:
                rhs=item[1]
                sum_pro+=item[2]
                if len(rhs)==2:
                    if rhs[0] not in nonterminals or rhs[1] not in nonterminals:
                        A=False
                elif len(rhs)==1:
                    if rhs[0] in nonterminals:
                        A=False
                else:
                    A=False
                   
            if math.isclose(sum_pro,1)!=True or A==False:
                return False
                break
        
        return True
       
if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
    if grammar.verify_grammar():
        print("The grammar is a valid PCFG in CNF.")
    else:
        print("ERROR: The grammar is not a valid PCFG in CNF.")
        