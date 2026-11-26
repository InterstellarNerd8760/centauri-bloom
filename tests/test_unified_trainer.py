import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from bloom.model import Bloom, Config
from bloom.trainer import collate_instruction, train_model
from bloom.trainer_data import RECIPES, arithmetic_rows, encode_pairs, encode_windows, instruction_pairs

class ToyTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[1+ord(c)%14 for c in text])

def test_all_dataset_recipes():
    assert set(RECIPES)=={"Jaron","FineWeb-Edu + TinyStories","FineWeb-Edu","TinyStories",
                         "Cosmopedia","Public-domain literature","Public-domain + TinyStories","Arithmetic instruction pilot"}
    assert RECIPES["Jaron"][0]=="instruction-tuning"
    assert RECIPES["FineWeb-Edu + TinyStories"][1]==("FineWeb-Edu","TinyStories")
    assert len(arithmetic_rows())==4512

def test_prompt_group_split_and_dedup():
    rows=[{"prompt":"HELLO?","response":"Hi"},{"prompt":" hello ","response":"Hi"},
          {"prompt":"Hello!","response":"Greetings"}]
    train,val=instruction_pairs(rows)
    assert len(train)+len(val)==2
    assert not train or not val

def test_instruction_masks_and_trim():
    tok=ToyTokenizer()
    ds=encode_pairs([("A","B"),("Longer","C")],tok,64,0)
    x,y=collate_instruction([ds[0],ds[1]])
    assert x.shape==y.shape
    assert x.shape[1]<64
    for a,b in zip(x,y):
        assert int((b!=-100).sum())==3 # leading space, answer, EOT
        assert b[b!=-100][-1]==0
        assert b[0]==-100

def test_windows():
    ds=encode_windows([[1,2,3,4],[5,6,7,8]],4,0)
    assert len(ds)==2
    assert ds[0][0].tolist()==[1,2,3,4]
    assert ds[0][1].tolist()==[2,3,4,0]

def test_tiny_training_counts():
    prior=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ds=TensorDataset(torch.tensor([[1,2,3,0],[1,2,4,0]]),
                         torch.tensor([[-100,3,0,-100],[-100,4,0,-100]]))
        loader=DataLoader(ds,batch_size=2)
        model=Bloom(Config(vocab_size=16,context=4,width=8,heads=2,layers=1))
        saved=[]
        report=train_model(model,loader,loader,{},"cpu",2,2,1e-3,47,
                           lambda *args:saved.append(args),lambda:{"test":"sample"},max_eval_batches=1)
        assert saved[-1][:4]==(2,24,16,2)
        assert report["evaluated_target_tokens"]==4
        assert report["candidate_val_loss"]>0
    finally:
        torch.set_num_threads(prior)

def test_notebook_defaults_and_pr_only():
    notebook=json.loads(Path("notebooks/bloom-trainer.ipynb").read_text())
    code=["".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"]
    assert len(code)==5
    for s in code:
        ast.parse(s)
    config=ast.parse(code[0])
    values={n.targets[0].id:n.value.value for n in config.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    assert values["DATASET"]=="Jaron"
    assert values["STEPS"]==3000
    assert "submit_candidate(" in code[-1]
    assert "upload_folder(" not in "\n".join(code)
