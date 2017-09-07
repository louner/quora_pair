from pycorenlp import StanfordCoreNLP
import json
from json import JSONEncoder, JSONDecoder
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import re
import logging
from preprocess import load_data
import os

embedding_size = 30

logging.basicConfig(filename='log/form_parse_tree.log', level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger('form_parse_tree')

#wordPat = re.compile(r'[a-zA-Z0-9\.\?\!_$-,]+')
wordPat = re.compile(r'[^\s\[\]\n]+')
spacePat = re.compile(r'[\s]+')

nlp = StanfordCoreNLP('http://localhost:9000')

word_ids = {}
delimeter = ' @ '

class WordNode:
    def __init__(self, nodeJson):
        if type(nodeJson) == str:
            word, word_id = nodeJson.split(delimeter)
            self.word = word
            self.word_id = int(word_id)
            self.leaf = True
        else:
            self.left = WordNode(nodeJson[0])
            self.right = WordNode(nodeJson[1])
            self.leaf = False

    def __repr__(self):
        if self.leaf:
            return '"%s"'%(self.word)
        else:
            return '[%s, %s]'%(repr(self.left), repr(self.right))

    def __getitem__(self, item):
        return self.__dict__[item]

def reformat(tree):
    if type(tree[1]) == str:
        word = tree[1]
        return word
    elif len(tree) == 2:
        return reformat(tree[1])

    else:
        return [reformat(tree[1]), reformat(['']+tree[2:])]

def transform_to_tree(tree_str):
    tree_str = tree_str.replace('(', '[').replace(')', ']').replace('\n', '')
    tree_str = wordPat.sub(lambda m: '"%s"'%(m.group(0)), tree_str)
    tree_str = spacePat.sub(',', tree_str)
    tree = json.loads(tree_str)
    return tree

def build_syntatic_tree(sentence):
    try:
        output = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})['sentences'][0]
        syntatic_tree = transform_to_tree(output['parse'])
        return reformat(syntatic_tree)

    except:
        logger.error(sentence)
        return None

def add_word_id(tree, word_ids):
    if type(tree) == str:
        if not tree in word_ids:
            word_ids[tree] = len(word_ids)
        return '%s%s%s'%(tree, delimeter, word_ids[tree])

    else:
        tree[0] = add_word_id(tree[0], word_ids)
        tree[1] = add_word_id(tree[1], word_ids)
        return tree

def add_word_id_files(files):
    if os.path.isfile('train/word_ids.json'):
        word_ids = json.load(open('train/word_ids.json'))
    else:
        word_ids = {}
    tmp_file = './tmp'

    for file in files:
        with open(file) as fin:
            with open(tmp_file, 'w') as fout:
                for line in fin:
                    tree = json.loads(line.strip('\n'))
                    tree = add_word_id(tree, word_ids)
                    fout.write('%s\n'%(json.dumps(tree)))
        os.rename(tmp_file, file)

    json.dump(word_ids, open('train/word_ids.json', 'w'))
    print(max(word_ids.values()))

if __name__ == '__main__':
    sentence = 'Why India does not apply the "Burma-Rohingya model" to deport illegal Bangladeshis?'
    tree = build_syntatic_tree(sentence)
    print(json.dumps(tree, indent=4))
    print(json.dumps(add_word_id(tree, {}), indent=4))
    r = WordNode(tree)
    print(r)


    '''
    sentence = 'How can I win her back ?'
    sentence = 'Peter and I like apples .'
    print(trace(build_lexical_tree(sentence)))
    '''