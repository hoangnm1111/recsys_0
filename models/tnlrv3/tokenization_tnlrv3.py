# coding=utf-8
"""Tokenization classes for TuringNLRv3."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

import transformers
if int(transformers.__version__[0]) <=3:
    from transformers.tokenization_bert import BertTokenizer, whitespace_tokenize
else:
    from transformers.models.bert.tokenization_bert import BertTokenizer, whitespace_tokenize


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
}


class TuringNLRv3Tokenizer(BertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)