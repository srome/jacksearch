# JackSearch

An image search tool that utilizes an LSTM (Long Short-Term Memory) neural network to caption images on a file system and then matches an input search phrase with the captions via word vector similarity. Results are ordered by relevance in returned in HTML format.

#### Who is Jack?

![Jack](jack.jpg?raw=true "Jack")

Jack was my family's cat for the last thirteen years. After a trip to the emergency room on new years eve 2016, Jack was diagnosed with heart failure and had to be put down. I had lots of images of Jack from over the years stored on several different external hard drives, and I wanted a way to search for them. I thought this was a good opportunity to work on a project in his memory.

## Technology Overview

The app is built on top of [TensorFlow](https://www.tensorflow.org/) and uses its command line interface functionality. The image caption generator that JackSearch uses is from the [im2txt](https://github.com/tensorflow/models/tree/master/im2txt) model implementation in TensorFlow. The natural language processing is accomplished via [spaCy](https://spacy.io/) and the [GloVe](http://nlp.stanford.edu/projects/glove/) word vectors.


## Install

These instructions explain how to install the repo.

### 1) System Prerequisites

* **Bazel** ([Install](http://bazel.io/docs/install.html)).

### 2) Python Installation

JackSearch has been tested on Python 3.5.2. This installation assumes you already have pip installed and I recommend you install the following in a virtual environment. Using pip, you may install the necessary packages via the requirements file:

```shell
pip install -r requirements.txt
```

Then, you have to install supplemental data for several packages:

* **Natural Language Toolkit (NLTK)**: Download 'all' available data ([details](http://www.nltk.org/data.html)).
* **SpaCy**: Install the parser and GloVE:
```shell
python -m spacy.en.download parser
python -m spacy.en.download glove
```

### 3) Run Example
'''shell
python main.py --base_dir=/path/to/your/directory \
 --search_phrase="Cats sitting on a bed" \
 --model_file=/path/to/your/model/checkpoint \
 --vocab_file=/path/to/your/vocab/file
'''
