"""Mock loader contract tests only; no model weights, CUDA allocation or HF load."""
import copy
import sys
from types import SimpleNamespace

import pytest

from exec_feedback.configuration import DEFAULT


@pytest.mark.parametrize('device,quantization', [('cuda', 'none'), ('cpu', 'none'), ('cuda', 'nf4')])
def test_single_device_loading_preserves_remote_streaming_change(monkeypatch, device, quantization):
    import torch
    import exec_feedback.models as module

    calls = {}
    model = SimpleNamespace(to=lambda dev: calls.update(moved=dev))
    def load(source, **kwargs):
        calls.update(source=source, kwargs=kwargs)
        return model
    def prepare(value, **kwargs):
        calls['prepared'] = kwargs
        return value
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        BitsAndBytesConfig=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(prepare_model_for_kbit_training=prepare))
    monkeypatch.setattr(module, 'load_tokenizer', lambda spec: 'offline-tokenizer-double')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    spec = copy.deepcopy(DEFAULT['model'])
    spec.update(device=device, quantization=quantization)
    loaded, _ = module.load_model(spec)
    assert loaded is model
    assert calls['kwargs']['trust_remote_code'] is False
    if device == 'cuda':
        assert calls['kwargs']['device_map'] == {'': 0}
        assert calls['kwargs']['low_cpu_mem_usage'] is True
        assert 'moved' not in calls
    else:
        assert 'device_map' not in calls['kwargs']
        assert calls['moved'] == 'cpu'
    if quantization == 'nf4':
        assert calls['kwargs']['quantization_config']['load_in_4bit'] is True
        assert calls['prepared']['use_gradient_checkpointing'] is False
