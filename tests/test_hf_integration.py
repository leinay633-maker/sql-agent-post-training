"""Offline HF/PEFT integration on a freshly initialized tiny Qwen3 architecture.

No pretrained weights or benchmark scores. Installs of the ml extra enable this
acceptance gate; missing ML packages are reported as SKIP, never PASS.
"""
import copy
import importlib.util
import json

import pytest

ML_AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ('transformers','peft','tokenizers','accelerate'))


@pytest.mark.skipif(not ML_AVAILABLE, reason='Install the ml extra to validate actual Transformers/PEFT integration')
def test_qwen_peft_train_save_reload_and_inference(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from exec_feedback.configuration import DEFAULT
    from exec_feedback.cli import make_fixture
    from exec_feedback.pipeline import build_sft
    from exec_feedback.core import write_jsonl
    from exec_feedback.train import fit
    from exec_feedback.models import TransformersPolicy
    torch.set_num_threads(1)
    torch.manual_seed(7)
    raw = Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'system':3,'user':4,'assistant':5,'SELECT':6,'finish':7},unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tiny=Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=2048,pad_token_id=1,eos_token_id=2))
    source=tmp_path/'tiny-qwen'
    tiny.save_pretrained(source); tokenizer.save_pretrained(source)
    task=make_fixture(tmp_path/'db')
    rows=build_sft([task],tmp_path/'db','success',allow_fixture=True)
    write_jsonl(tmp_path/'data.jsonl',rows)
    spec={**copy.deepcopy(DEFAULT['model']), 'local_path':str(source), 'id':str(source), 'device':'cpu','quantization':'none','max_context_tokens':2048,'max_tokens':16}
    settings={**DEFAULT['training'],'max_steps':2,'max_length':2048,'batch_size':1,'gradient_accumulation_steps':1,'lora_r':2,'lora_alpha':4,'save_steps':1,'max_seconds':120}
    result=fit(tmp_path/'data.jsonl',spec,settings,tmp_path/'training',seed=42,allow_fixture=True)
    assert result['training_completed'] and result['adapter_reload_verified'] and result['fixture']
    assert result['metrics']['step']==2
    policy=TransformersPolicy(spec,tmp_path/'training/adapter')
    try:
        text=policy([{'role':'user','content':'SELECT'}])
        assert isinstance(text,str)
        assert policy.metadata['usage']['completion_tokens']>0
        batch = policy.generate_batch([[{'role':'user','content':'SELECT'}],
                                       [{'role':'user','content':'SELECT SELECT SELECT'}]])
        assert len(batch)==2 and all(isinstance(row['text'],str) for row in batch)
    finally: policy.close()
