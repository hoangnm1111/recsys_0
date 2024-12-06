from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

TuringNLRv3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
}


class TuringNLRv3Config(PretrainedConfig):
    pretrained_config_archive_map = TuringNLRv3_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=28996,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=6,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 source_type_id=0,
                 target_type_id=1,
                 **kwargs):
        super(TuringNLRv3Config, self).__init__(**kwargs)
        if isinstance(vocab_size, str) or (sys.version_info[0] == 2
                                           and isinstance(vocab_size, unicode)):
            with open(vocab_size, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.source_type_id = source_type_id
            self.target_type_id = target_type_id
        else:
            raise ValueError("First argument must be either a vocabulary size (int)")
