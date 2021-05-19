import logging

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from .roberta import Roberta
from nlpaug.util.selection.filtering import *


class XLMRoberta(Roberta):
    START_TOKEN = '<s>'
    SEPARATOR_TOKEN = '</s>'
    MASK_TOKEN = '<mask>'
    PAD_TOKEN = '<pad>'
    UNKNOWN_TOKEN = '<unk>'
    SUBWORD_PREFIX = '▁'
