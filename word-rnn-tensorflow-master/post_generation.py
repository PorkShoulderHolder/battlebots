import sys
import re
from nltk.corpus import cmudict

syl_dict = cmudict.dict()


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


def uniform_syl(lines, num=11):
    out = ""
    lines_copy = lines
    while len(out) < len(lines):
        new_line = syllables(lines_copy, num=num)
        lines_copy = lines_copy[len(new_line):]
        out += new_line + '<br>'

    return out
