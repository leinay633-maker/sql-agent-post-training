"""Real CPU tensor/LoRA optimization; no pretrained-model quality is asserted."""
import copy
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from exec_feedback.optimization import batch_tensors, completion_loss, train_engine
from exec_feedback.storage import read_json


class TinyAdapterLM(torch.nn.Module):
    """Frozen backbone + actual low-rank A/B update for integration testing."""
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(17,8)
        self.base=torch.nn.Linear(8,17,bias=False)
        self.embedding.requires_grad_(False)
        self.base.requires_grad_(False)
        self.lora_A=torch.nn.Parameter(torch.randn(2,8)*.1)
        self.lora_B=torch.nn.Parameter(torch.zeros(17,2))
        self.drop=torch.nn.Dropout(.1)
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        hidden=self.embedding(input_ids)
        logits=self.base(hidden)+(self.drop(hidden)@self.lora_A.T)@self.lora_B.T
        return SimpleNamespace(logits=logits)


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    base=TinyAdapterLM()
    examples=[{"input_ids":[1,2,3,4,4],"attention_mask":[1]*5,"labels":[-100,-100,-100,4,4]},
              {"input_ids":[1,3,4],"attention_mask":[1]*3,"labels":[-100,-100,4]}]
    settings={"max_steps":6,"gradient_accumulation_steps":2,"batch_size":1,"learning_rate":.05,
              "save_steps":1,"max_seconds":60,"warmup_ratio":0}
    return base,examples,settings


def test_masked_token_loss_matches_manual_formula(setup):
    model,examples,_=setup
    model.eval()
    batch=batch_tensors(examples,0,"cpu")
    loss,n=completion_loss(model,batch)
    logits=model(batch["input_ids"]).logits
    labels=batch["labels"][:,1:]
    mask=labels!=-100
    expected=torch.nn.functional.cross_entropy(logits[:,:-1][mask],labels[mask],reduction="sum")
    assert n==3 and torch.allclose(loss,expected)
    assert batch["labels"][1,-1].item()==-100


def test_real_lora_training_and_exact_checkpoint_resume(setup,tmp_path):
    base,examples,settings=setup
    initial=copy.deepcopy(base.state_dict())
    full=TinyAdapterLM(); full.load_state_dict(initial)
    result=train_engine(full,examples,settings,tmp_path/"full",pad_id=0,seed=42,identity="cpu-test")
    assert result["completed"] and math.isfinite(result["loss"])
    assert not torch.equal(full.lora_B,initial["lora_B"])
    assert torch.equal(full.base.weight,initial["base.weight"])
    partial=TinyAdapterLM(); partial.load_state_dict(initial)
    stopped=train_engine(partial,examples,settings,tmp_path/"resume",pad_id=0,seed=42,identity="cpu-test",stop_after_steps=3)
    assert not stopped["completed"] and stopped["step"]==3
    fresh=TinyAdapterLM(); fresh.load_state_dict(initial)
    resumed=train_engine(fresh,examples,settings,tmp_path/"resume",pad_id=0,seed=42,identity="cpu-test")
    assert resumed["completed"]
    for name,p in full.named_parameters():
        assert torch.equal(p,dict(fresh.named_parameters())[name]),name
    assert read_json(tmp_path/"resume/state.json")["completed"]


def test_modified_training_inputs_cannot_resume(setup,tmp_path):
    model,examples,settings=setup
    train_engine(model,examples,settings,tmp_path/"out",pad_id=0,seed=42,identity="one",stop_after_steps=1)
    with pytest.raises(ValueError,match="changed"):
        train_engine(model,examples,settings,tmp_path/"out",pad_id=0,seed=42,identity="two")


def test_time_budget_does_not_claim_completion(setup,tmp_path,monkeypatch):
    model,examples,settings=setup
    clock = iter([100.0] + [101.0] * 30)
    monkeypatch.setattr("exec_feedback.optimization.time.monotonic", lambda: next(clock))
    result=train_engine(model,examples,{**settings,"max_seconds":0.5},tmp_path/"stop",pad_id=0,seed=42,identity="stop")
    assert not result["completed"] and result["step"]==0
    assert list((tmp_path/"stop").glob("checkpoint-*/complete.json"))


def test_no_supervision_and_nonfinite_loss_rejected(setup,tmp_path):
    model,examples,settings=setup
    broken=copy.deepcopy(examples)
    broken[0]["labels"]=[-100]*5
    with pytest.raises(ValueError,match="supervise"):
        train_engine(model,broken,settings,tmp_path/"bad",pad_id=0,seed=42,identity="bad")
    with torch.no_grad(): model.lora_B.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        train_engine(model,examples,settings,tmp_path/"nan",pad_id=0,seed=42,identity="nan")
    assert not read_json(tmp_path/"nan/state.json")["completed"]


def test_corrupt_checkpoint_is_rejected_before_torch_load(setup,tmp_path):
    model,examples,settings=setup
    train_engine(model,examples,settings,tmp_path/'out',pad_id=0,seed=42,identity='hash',stop_after_steps=1)
    target=tmp_path/'out/checkpoint-00000001/trainable.pt'
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        train_engine(model,examples,settings,tmp_path/'out',pad_id=0,seed=42,identity='hash')


def test_unsaved_metric_tail_is_preserved_and_not_double_counted(setup,tmp_path):
    from exec_feedback.storage import append_record,recover_records
    model,examples,settings=setup
    out=tmp_path/'out'
    train_engine(model,examples,settings,out,pad_id=0,seed=42,identity='tail',stop_after_steps=1)
    append_record(out/'metrics.jsonl',{'step':2,'loss':99,'not_checkpointed':True})
    train_engine(model,examples,settings,out,pad_id=0,seed=42,identity='tail')
    assert [x['step'] for x in recover_records(out/'metrics.jsonl')]==list(range(1,7))
    assert list(out.glob('metrics.rolled-back-*.jsonl'))
