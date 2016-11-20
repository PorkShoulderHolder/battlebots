import sys
import re


def clean_line(line):
    return re.sub("\[(.*?)\]", "", line)


def clean_file(fn):
    with open(fn) as f:
        out = ""
        for line in f.readlines():
            out += clean_line(line)

