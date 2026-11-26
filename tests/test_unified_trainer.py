import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from bloom.model import Bloom, Config
from bloom.trainer import CHAT_REPOS, TEXT_REPOS, collate_instruction, target_repo, train_model
from bloom.trainer_data import (
    INSTRUCT_MIX, INSTRUCT_SOURCES, JARON_MIX_WEIGHT, RECIPES, RECIPE_BUDGETS,
    arithmetic_rows, convert_instruct_source, dolly_pair, encode_pairs, encode_windows,
    filter_pairs, fits_instruction, gsm8k_pair, instruction_pairs, needs_jaron,
    oasst_pairs, prepare, prepare_instruct_mix, weighted_mix,
)

class ToyTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[1+ord(c)%14 for c in text])

def test_all_dataset_recipes():
    assert set(RECIPES)=={"Jaron","FineWeb-Edu + TinyStories","FineWeb-Edu","TinyStories",
                         "Cosmopedia","Public-domain literature","Public-domain + TinyStories",
                         "FineWeb-Edu + literature + TinyStories","Arithmetic instruction pilot",
                         INSTRUCT_MIX}
    assert RECIPES["FineWeb-Edu + literature + TinyStories"][1]==("FineWeb-Edu","Public-domain literature","TinyStories")
    assert RECIPE_BUDGETS["FineWeb-Edu + literature + TinyStories"]=={"FineWeb-Edu":20_000_000,"TinyStories":500_000}
    assert RECIPES["Jaron"][0]==RECIPES[INSTRUCT_MIX][0]=="instruction-tuning"
    assert RECIPES["FineWeb-Edu + TinyStories"][1]==("FineWeb-Edu","TinyStories")
    assert needs_jaron("Jaron") and needs_jaron(INSTRUCT_MIX) and not needs_jaron("TinyStories")
    assert target_repo("50M", INSTRUCT_MIX)==CHAT_REPOS["50M"]=="Centauri-Bloom/Bloom-Chat-0.2"
    assert target_repo("50M", "Jaron")=="Centauri-Bloom/Bloom-Chat-0.2"
    assert target_repo("50M", "FineWeb-Edu + literature + TinyStories")==TEXT_REPOS["50M"]
    assert target_repo("14M", "Jaron")==TEXT_REPOS["14M"]
    assert target_repo("14M", INSTRUCT_MIX)==CHAT_REPOS["14M"]
    assert INSTRUCT_SOURCES["Dolly-15k"]["weight"]==2
    assert INSTRUCT_SOURCES["OpenAssistant OASST1"]["weight"]==2
    assert INSTRUCT_SOURCES["GSM8K"]["weight"]==1
    assert JARON_MIX_WEIGHT==4
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
    assert values["MODEL"]=="14M"
    assert values["DATASET"]=="Jaron"
    assert values["STEPS"]==3000
    assert "FineWeb-Edu + literature + TinyStories" in code[0]
    assert INSTRUCT_MIX in code[0]
    assert "PARENT_RUN_DIR" in code[0]
    assert "TARGET_REPOS" in "\n".join(code)
    assert "target_repo(" in "\n".join(code)
    assert "Bloom-Text-0.2" in "\n".join(code)
    assert "Bloom-Chat-0.2" in "\n".join(code)
    assert "parent_run_dir" in "\n".join(code)
    assert "needs_jaron(" in "\n".join(code)
    assert "pick_device(" in "\n".join(code)
    assert "submit_candidate(" in code[-1]
    assert "upload_folder(" not in "\n".join(code)
    assert "os.environ.get(\"HF_TOKEN\")" in "\n".join(code)
    assert "create_repo(" not in "\n".join(code)

def test_50m_config_param_count_and_cpu_step():
    from bloom.trainer import CONFIG_50M, PARAM_COUNT_50M, load_model_config
    cfg = Config(**load_model_config(CONFIG_50M))
    assert (cfg.vocab_size, cfg.context, cfg.width, cfg.heads, cfg.layers) == (8192, 512, 640, 10, 10)
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = Bloom(cfg)
        assert sum(p.numel() for p in model.parameters()) == PARAM_COUNT_50M == 54_807_040
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
    finally:
        torch.set_num_threads(prior)

def test_resume_continues_from_saved_step(tmp_path):
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ds = TensorDataset(torch.tensor([[1, 2, 3, 0], [1, 2, 4, 0]]),
                           torch.tensor([[-100, 3, 0, -100], [-100, 4, 0, -100]]))
        loader = DataLoader(ds, batch_size=2)
        model = Bloom(Config(vocab_size=16, context=4, width=8, heads=2, layers=1))
        opt_path = tmp_path / "partial-optimizer.pt"
        saved = []
        train_model(model, loader, loader, {}, "cpu", 2, 1, 1e-3, 47,
                    lambda *args: saved.append(args), lambda: {"test": "sample"},
                    max_eval_batches=1, optimizer_file=str(opt_path))
        assert saved[-1][0] == 2
        assert opt_path.is_file()
        saved.clear()
        train_model(model, loader, loader, {}, "cpu", 4, 1, 1e-3, 47,
                    lambda *args: saved.append(args), lambda: {"test": "sample"},
                    max_eval_batches=1, start_step=2, optimizer_file=str(opt_path),
                    token_counts=(8, 4, 2))
        assert saved[0][0] == 4
        assert saved[0][1] > 8
    finally:
        torch.set_num_threads(prior)

def test_open_instruction_converters_and_filters():
    tok = ToyTokenizer()
    dolly = dolly_pair({"instruction":"Name a color.","context":"","response":"Blue."})
    dolly_ctx = dolly_pair({"instruction":"What color?","context":"The sky is blue.","response":"Blue."})
    assert dolly=={"prompt":"Name a color.","response":"Blue."}
    assert dolly_ctx["prompt"].startswith("The sky is blue.")
    assert gsm8k_pair({"question":"What is 2+2?","answer":"4"})=={"prompt":"What is 2+2?","response":"4"}
    assert dolly_pair({"instruction":"","response":"x"}) is None
    root = {"message_id":"p","parent_id":None,"role":"prompter","lang":"en","text":"What is Earth?",
            "deleted":False,"review_result":True,"labels":{"name":["pii","spam"],"value":[0,0]}}
    good = {"message_id":"a","parent_id":"p","role":"assistant","lang":"en","text":"A planet.","rank":0,
            "deleted":False,"review_result":True,
            "labels":{"name":["pii","spam","quality","hate_speech","sexual_content","not_appropriate"],"value":[0,0,0.9,0,0,0]}}
    pii = dict(good, message_id="b", labels={"name":["pii","quality"],"value":[0.5,0.9]})
    ranked = dict(good, message_id="c", rank=1)
    follow = dict(good, message_id="d", parent_id="a")
    pairs = oasst_pairs([root, good, pii, ranked, follow])
    assert pairs==[{"prompt":"What is Earth?","response":"A planet."}]
    converted = convert_instruct_source("Dolly-15k", [{"instruction":"Hi","context":"","response":"Hello"}])
    assert converted[0]["prompt"]=="Hi"
    assert fits_instruction("A","B",tok,64,0)
    assert not fits_instruction("A","B",tok,2,0)
    kept = filter_pairs([{"prompt":"A","response":"B"},{"prompt":"","response":"B"}], tok, 64, 0)
    assert kept==[{"prompt":"A","response":"B"}]

def test_weighted_mix_upweights_jaron():
    jaron = [("j"+str(i), "a") for i in range(10)]
    dolly = [("d"+str(i), "a") for i in range(40)]
    oasst = [("o"+str(i), "a") for i in range(40)]
    gsm = [("g"+str(i), "a") for i in range(20)]
    mixed, used = weighted_mix(
        {"Jaron":jaron,"Dolly-15k":dolly,"OpenAssistant OASST1":oasst,"GSM8K":gsm},
        {"Jaron":4,"Dolly-15k":2,"OpenAssistant OASST1":2,"GSM8K":1},
        90, seed=47, max_repeat=8)
    assert used["Jaron"]>=used["GSM8K"]
    assert used["Jaron"]>=10
    assert sum(used.values())==len(mixed)<=90
    # Jaron examples are repeated to raise their share.
    assert used["Jaron"]>len(jaron)

def test_instruct_mix_prepare_smoke(tmp_path, monkeypatch):
    tok = ToyTokenizer()
    def fake_jaron(_path):
        # Q15? and Q24? hash into the 10% validation split; the rest are train.
        return [{"prompt":q,"response":"A."} for q in
                ("Q0?","Q1?","Q2?","Q3?","Q4?","Q5?","Q15?","Q24?")]
    def fake_source(name, tok, context, eot):
        rows = {
            "Dolly-15k":[{"prompt":q,"response":"A."} for q in ("Q6?","Q7?","Q26?")],
            "OpenAssistant OASST1":[{"prompt":q,"response":"A."} for q in ("Q8?","Q9?","Q71?")],
            "GSM8K":[{"prompt":q,"response":"A."} for q in ("Q10?","Q72?")],
        }[name]
        meta = {"name":name,"dataset":"stub/"+name,"license_note":"test","filtered_count":len(rows),
                "cap":INSTRUCT_SOURCES[name]["cap"],"weight":INSTRUCT_SOURCES[name]["weight"]}
        return rows, meta
    monkeypatch.setattr("bloom.trainer_data.load_jaron_rows", fake_jaron)
    t,v,sources,meta = prepare_instruct_mix(tok, 64, 0, 47, tmp_path/"unused.json", source_loader=fake_source)
    assert len(t)>0 and len(v)>0
    assert set(sources)>= {"Jaron","Dolly-15k","OpenAssistant OASST1","GSM8K"}
    names = [s["name"] for s in meta["sources"]]
    assert names==["Jaron","Dolly-15k","OpenAssistant OASST1","GSM8K"]
    jaron_meta = meta["sources"][0]
    assert jaron_meta["weight"]==4
    assert "not uploaded" in jaron_meta["note"]
    # Loss mask still covers assistant tokens only.
    x,y = t[0]
    assert int((y!=-100).sum())>=2
    assert y[0]==-100
    assert meta["prompt_format"].startswith("User:")
    assert "raw" not in str(meta).lower() or "not uploaded" in str(meta)

def test_instruct_mix_requires_jaron():
    tok = ToyTokenizer()
    try:
        prepare(INSTRUCT_MIX, ".", tok, 64, 0, 47, jaron_path=None)
    except ValueError as exc:
        assert "Jaron" in str(exc)
    else:
        raise AssertionError("expected missing Jaron path to fail")
