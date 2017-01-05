# Copyright (C) 2016 Scott Rome. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Turn off tensor flow warnings
from im2txt import configuration, inference_wrapper
from im2txt.inference_utils import caption_generator, vocabulary
from results import render_results
from search import Searcher
import logging, os, webbrowser



# Define tensor flow command line flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("search_phrase", "cat",
                       "Phrase to search for in image captions.")
tf.flags.DEFINE_string("base_dir", None,
                       "Base directory to search in (includes subdirectories).")
tf.flags.DEFINE_string("model_file", "",
                       "Locaion of the model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Locaion of the vocabulary file.")

def main(_):
    # Import config
    import yaml

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    # Load NLP libraries
    logger.info('Loading NLP library')
    import spacy
    from nltk.corpus import stopwords
    nlp = spacy.load('en')
    STOP_WORDS = set(stopwords.words('english'))

    # Parse search phrase
    search_input = FLAGS.search_phrase
    search_phrase = nlp(' '.join([word for word in search_input.split(' ') if word not in STOP_WORDS]))
    logger.info('Search phrase: "%s"' % search_phrase.text)

    results = []

    # Required for model
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.model_file)
    g.finalize()
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    # Find files to search
    search_dir = FLAGS.base_dir if FLAGS.base_dir is not None else os.path.dirname(os.path.abspath(__file__))
    files = Searcher.search_from_dir(search_dir)
    num_files = len(files)
    logger.info('%d file(s) found' % num_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint and instantiate caption generator model.
        restore_fn(sess)
        generator = caption_generator.CaptionGenerator(model, vocab)

        # Caption the files
        count = 0
        for file_path in files:
            count+=1.
            try:
                with tf.gfile.GFile(file_path, "r") as f:
                  image = f.read()
                captions = generator.beam_search(sess, image)
                logger.info("Captioning image %f: %s" % (count/num_files,file_path))
                best_caption = captions[0] # Just take the most probable caption
                sentence = nlp(" ".join([vocab.id_to_word(word) for word in best_caption.sentence[1:-1] if word not in STOP_WORDS]))
                results.append((file_path, sentence.text, search_phrase.similarity(sentence)))
            except Exception as e:
                logger.warning('Failed to caption image: %s' % file_path)
                  

        render_results(search_phrase.text, sorted(results, key= lambda x : x[2], reverse=True))

        webbrowser.open('output.html',new=2)

if __name__ == "__main__":
    tf.app.run()

