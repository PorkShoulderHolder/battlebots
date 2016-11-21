import sys
import re
from nltk.corpus import cmudict, brown, movie_reviews, treebank
import pronouncing as prn
from utils import TextLoader
syl_dict = cmudict.dict()
from gensim.models import Word2Vec




def get_word2vec(train_fn="data/rap/input.txt", saved_model_fn="save/save/w2v"):
    try:
        print "loading word2vec model at {0}".format(saved_model_fn)
        return Word2Vec.load(saved_model_fn)
    except IOError:
        print "no word2vec model found at {0}".format(saved_model_fn)
        with open(train_fn) as f:
            data = f.read()
            clean = TextLoader.clean_str(data)
            lines = clean.split('\n')
            full_data = lines + brown.sents() + movie_reviews.sents() + treebank.sents()
            print "training word2vec model"
            model = Word2Vec(full_data)
            model.save(saved_model_fn)
            return model

w2v = get_word2vec()


def vowel_counter(word):
    num_vowels = 0
    for char in word:
        if char in "aeiouAEIOU":
            num_vowels += 1
    return num_vowels


def syllables(line, num=11):
    total = 0
    output = ''
    for word in line.split(" "):
        try:
            sylls = syl_dict[word.lower()]
            for x in sylls:
                total += len(list(y for y in x if y[-1].isdigit()))
        except KeyError:
            total += vowel_counter(word)
        output += " " + word
        if total >= num:
            break
    return output


def clean_msg(line):
    new_line = re.sub("\[(.*?)\]", "", line)
    new_line = new_line.replace("(", "").replace(")", "")
    new_line = re.sub('[^a-zA-Z\s]', '', new_line)
    return new_line


def rhyme_candidates(lines, freq=6):
    words = lines.split(" ")
    candidates = words[::freq]
    if len(candidates) % 2 == 1:
        candidates = candidates[1:]
    return candidates, words


def replace_w_rhymes(lines, freq=6):
    candidates, words = rhyme_candidates(lines, freq=freq)
    rhymed = make_pairs(candidates)
    for i, word in enumerate(rhymed):
        words[i * 6] = word
    return words

    
def get_best_rhyme(first, second):

    f_rhymes = prn.rhymes(first)
    scores_f = [(w2v.similarity(r, second), r) for r in f_rhymes]
    best_candidate_f = max(scores_f, lambda x: x[0])
    s_rhymes = prn.rhymes(second)
    scores_s = [(w2v.similarity(r, first), r) for r in s_rhymes]
    best_candidate_s = max(scores_s, lambda x: x[0])
    if best_candidate_f[0] > best_candidate_s[0]:
        return best_candidate_f[1], second
    else:
        return first, best_candidate_s[1]


def make_pairs(words):
    i = 0
    while i < len(words):
        first, second = get_best_rhyme(words[i], words[i + 1])
        words[i] = first
        words[i + 1] = second
    return words


def uniform_syl(lines, num=21):
    out = ""
    lines_copy = lines
    while len(out) < len(lines):
        new_line = syllables(lines_copy, num=num)
        lines_copy = lines_copy[len(new_line):]
        out += new_line + '<br>'

    return out
