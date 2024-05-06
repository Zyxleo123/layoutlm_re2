from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class NEROutput(TokenClassifierOutput):
    word_lengths: Optional[List] = None
    word_ids: Optional[List] = None
