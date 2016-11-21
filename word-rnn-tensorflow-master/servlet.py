from flask import Flask, Request, request, render_template, send_file
import sys
from sample import sample
from model import Model
import tensorflow as tf
import json
import cPickle
import os
from argparse import Namespace
from post_generation import clean_msg, uniform_syl

app = Flask(__name__)
MODEL_DIRS = ["save/save"]
MODEL_LOOKUP = {}
sess = None


def load_models():
    global sess
    if sess is None:
        sess = tf.Session()
    for name in MODEL_DIRS:
        with open(os.path.join(name, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(name, 'words_vocab.pkl'), 'rb') as f:
            words, vocab = cPickle.load(f)
        model = Model(saved_args, True)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(name)
        MODEL_LOOKUP[name] = {"model": model, "words": words, "vocab": vocab, "ckpt": ckpt}


@app.route('/prompt')
def prompt():
    prompt = request.args.get('text')
    challenger = request.args.get('challenger')
    n = int(request.args.get('n'))
    args = Namespace(prime=prompt, n=n, save_dir="save/save", sample=1)
    output = sample(args)
    new_example = output[0]
    for character in output[1:]:
        # Append an underscore if the character is uppercase.
        if character.isupper():
            new_example += '\n'
        new_example += character
    return json.dumps({"text": new_example})


@app.route('/raw')
def gen_text():
    prompt = request.args.get('text')
    challenger = request.args.get('challenger')
    n = int(request.args.get('n'))
    args = Namespace(prime=prompt, n=n, save_dir="save/save", sample=1)
    output = sample(args)
    output = clean_msg(output)
    output = uniform_syl(output)
    print output
    return output

if sys.platform == "darwin":
    app.run(port=int(sys.argv[1]), host='0.0.0.0', debug=False)
else:
    app.run(port=int(sys.argv[1]), host='0.0.0.0')
