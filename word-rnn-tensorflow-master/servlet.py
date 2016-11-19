from flask import Flask, Request, request, render_template, send_file
import sys
from sample import sample
from model import Model
import tensorflow as tf
import json
import cPickle
import os
from argparse import Namespace

app = Flask(__name__)
MODEL_DIRS = ["save"]
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
def process_url():
    prompt = request.args.get('text')
    challenger = request.args.get('challenger')
    args = Namespace(prime=prompt, n=200, save_dir=challenger, sample=1)
    output = sample(args)
    return json.dumps({"text": output})




app.run(port=8888, host='0.0.0.0')

