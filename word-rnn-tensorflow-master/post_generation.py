import sys
import re
from nltk.corpus import cmudict, brown, movie_reviews, treebank
import pronouncing as prn
from utils import TextLoader
syl_dict = cmudict.dict()
from gensim.models import Word2Vec
from random import shuffle
from progressbar import ProgressBar


def get_word2vec(train_fn="data/rap/input.txt", saved_model_fn="save/save/GoogleNews-vectors-negative300.bin"):
    try:
        print "loading word2vec model at {0}".format(saved_model_fn)
        model = Word2Vec.load_word2vec_format(saved_model_fn, binary=True)
        print "model loaded"
        return model
    except IOError:
        print "no word2vec model found at {0}".format(saved_model_fn)
        with open(train_fn) as f:
            data = f.read()
            clean = TextLoader.clean_str(data)
            lines = [line.split(" ") for line in clean.split('\n')]
            full_data = brown.sents() + movie_reviews.sents() + treebank.sents() + lines
            print "training word2vec model"
            model = Word2Vec(workers=8)
            model.build_vocab(full_data)
            for i in xrange(0, 5):
                print "epoch " + str(i + 1)
                # full_data = shuffle(full_data)
                pb = ProgressBar(maxval=len(full_data))
                chunk_size = len(full_data) / 100
                j = 0
                pb.start()
                while j + chunk_size < len(full_data):
                    model.train(full_data[j: j + chunk_size])
                    j += chunk_size
                    pb.update(j)

            print "done training"
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
    print rhymed
    for i, word in enumerate(rhymed):
        words[i * freq] = word
    return " ".join(words)


def get_similarities(word, rhymes):
    scores = []
    for r in rhymes:
        try:
            scores.append((w2v.similarity(r, word), r))
        except KeyError:
            pass
    return scores


def get_best_rhyme(first, second):

    f_rhymes = prn.rhymes(first)
    scores_f = get_similarities(second, f_rhymes)
    s_rhymes = prn.rhymes(second)
    scores_s = get_similarities(first, s_rhymes)

    f_active = len(scores_f) > 0 and len(scores_s) == 0
    both_active = len(scores_f) > 0 and len(scores_s) > 0
    s_active = len(scores_s) > 0 and len(scores_f) == 0
    if s_active:
        best_candidate_s = max(scores_s, key=lambda x: x[0])
    if f_active:
        best_candidate_f = max(scores_f, key=lambda x: x[0])
    if both_active:
        best_candidate_s = max(scores_s, key=lambda x: x[0])
        best_candidate_f = max(scores_f, key=lambda x: x[0])

    if f_active or (both_active and best_candidate_f[0] > best_candidate_s[0]):
        return first, best_candidate_f[1]
    elif s_active or (both_active and best_candidate_f[0] <= best_candidate_s[0]):
        return best_candidate_s[1], second
    else:
        return first, second


def make_pairs(words):
    i = 0
    while i < len(words):
        first, second = get_best_rhyme(words[i], words[i + 1])
        words[i] = first
        words[i + 1] = second
        i += 2
    return words


def uniform_syl(lines, num=21):
    out = ""
    lines_copy = lines
    while len(out) < len(lines):
        new_line = syllables(lines_copy, num=num)
        lines_copy = lines_copy[len(new_line):]
        out += new_line + '<br>'

    return out
