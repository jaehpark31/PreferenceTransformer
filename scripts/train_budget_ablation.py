#!/usr/bin/env python
"""Headless AntMaze *human-label* budget comparison: MR vs NMR vs PT.

Uses the released PreferenceTransformer human_label pickles as pairwise
preferences. Human labels are converted as:
  label 0  -> first segment preferred
  label 1  -> second segment preferred
  label -1 -> tie / uncertain (0.5 target)

Reward models are trained from pairwise preferences, then full D4RL transitions
are relabeled and IQL is trained/evaluated.
"""
import os, sys, time, json, csv, random, pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# ---- Runtime env for headless MuJoCo/D4RL ----
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

# Prefer the active conda/virtualenv instead of hard-coded machine paths.
ENV_PREFIX = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
if ENV_PREFIX:
    os.environ["PATH"] = f"{ENV_PREFIX}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([
        f"{ENV_PREFIX}/lib", os.environ.get("LD_LIBRARY_PATH", "")
    ])
    os.environ["CPATH"] = f"{ENV_PREFIX}/include:" + os.environ.get("CPATH", "")
    os.environ["LIBRARY_PATH"] = f"{ENV_PREFIX}/lib:" + os.environ.get("LIBRARY_PATH", "")

# If the local MuJoCo 2.1 install exists, use it; otherwise rely on the user's environment.
default_mujoco = Path.home() / ".mujoco" / "mujoco210"
if "MUJOCO_PY_MUJOCO_PATH" not in os.environ and default_mujoco.exists():
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(default_mujoco)
    os.environ["LD_LIBRARY_PATH"] = ":".join([
        str(default_mujoco / "bin"), os.environ.get("LD_LIBRARY_PATH", "")
    ])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm.auto import trange

# ---- Config ----
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
LABEL_SOURCE = os.environ.get("LABEL_SOURCE", "human").lower()
DEFAULT_BUDGETS = "333,666,1000" if LABEL_SOURCE == "human" else "10000,5000,2500"
DEFAULT_SEGMENT_LEN = "100" if LABEL_SOURCE == "human" else "16"
RUN_TAG = os.environ.get("RUN_TAG") or "human_label_budget_mr_nmr_pt_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_DIR / "runs" / RUN_TAG
RUN_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = {
    "env_name": os.environ.get("ENV_NAME", "antmaze-medium-play-v2"),
    "label_source": LABEL_SOURCE,
    "human_label_dir": os.environ.get("HUMAN_LABEL_DIR", str(PROJECT_DIR / "PreferenceTransformer" / "human_label")),
    "human_budget_split": os.environ.get("HUMAN_BUDGET_SPLIT", "nested_shuffle"),
    "data_seed": int(os.environ.get("DATA_SEED", "3407")),
    "models": [m.strip().lower() for m in os.environ.get("MODELS", "mr,nmr,pt").split(",") if m.strip()],
    "budgets": [int(x) for x in os.environ.get("BUDGETS", DEFAULT_BUDGETS).split(",") if x.strip()],
    "segment_len": int(os.environ.get("SEGMENT_LEN", DEFAULT_SEGMENT_LEN)),
    "reward_steps": int(os.environ.get("REWARD_STEPS", "20000")),
    "iql_steps": int(os.environ.get("IQL_STEPS", "100000")),
    "eval_episodes": int(os.environ.get("EVAL_EPISODES", "10")),
    "log_every": int(os.environ.get("LOG_EVERY", "5000")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "256")),
    "pref_batch_size": int(os.environ.get("PREF_BATCH_SIZE", "64")),
    "hidden_dims": [256, 256],
    "seq_hidden": int(os.environ.get("SEQ_HIDDEN", "128")),
    "pt_layers": int(os.environ.get("PT_LAYERS", "2")),
    "pt_heads": int(os.environ.get("PT_HEADS", "4")),
    "discount": 0.99,
    "expectile": 0.7,
    "temperature": 3.0,
    "max_adv_weight": 100.0,
    "tau": 0.005,
    "lr": 3e-4,
    "reward_lr": 3e-4,
    "antmaze_reward_shift": True,
    "seed": int(os.environ.get("SEED", "0")),
    "run_oracle": os.environ.get("RUN_ORACLE", "0") == "1",
    "informative_pair_frac": float(os.environ.get("INFORMATIVE_PAIR_FRAC", "0.8")),
    # D4RL AntMaze marks success/goal transitions as terminal repeatedly; those
    # are not reliable trajectory boundaries for building preference segments.
    # Auto mode prefers next_obs -> next row discontinuities, which recovers the
    # original 1000-step AntMaze trajectories.
    "episode_boundary_mode": os.environ.get("EPISODE_BOUNDARY_MODE", "auto"),
    "save_rollouts": os.environ.get("SAVE_ROLLOUTS", "0") == "1",
    "save_checkpoints": os.environ.get("SAVE_CHECKPOINTS", "0") == "1",
    "rollout_episodes": int(os.environ.get("ROLLOUT_EPISODES", "1")),
    "rollout_max_steps": int(os.environ.get("ROLLOUT_MAX_STEPS", "1000")),
}
print("RUN_DIR", RUN_DIR, flush=True)
print("CONFIG", json.dumps(CONFIG, indent=2), flush=True)

# ---- Utils ----
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def as_f32(x): return np.asarray(x, dtype=np.float32)
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers=[]
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1]), (activation if j < len(sizes)-2 else output_activation)()]
    return nn.Sequential(*layers)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(CONFIG["seed"])
print("Python", sys.version.split()[0], "Torch", torch.__version__, "device", device, flush=True)
if torch.cuda.is_available(): print("GPU", torch.cuda.get_device_name(0), flush=True)

# ---- Dataset ----
def load_d4rl(env_name: str):
    import gym, d4rl  # noqa
    env = gym.make(env_name)
    ds = d4rl.qlearning_dataset(env)
    data = {k: as_f32(ds[k]) for k in ["observations", "actions", "rewards", "next_observations", "terminals"]}
    data["rewards"] = data["rewards"].reshape(-1)
    data["terminals"] = data["terminals"].reshape(-1)
    data["timeouts"] = as_f32(ds.get("timeouts", np.zeros_like(data["terminals"]))).reshape(-1)
    return data, env

def episode_slices(data):
    terminals=np.asarray(data["terminals"]).reshape(-1)
    timeouts=np.asarray(data.get("timeouts", np.zeros_like(terminals))).reshape(-1)
    mode=CONFIG.get("episode_boundary_mode", "auto")

    # Prefer true dataset-row discontinuities when available. In D4RL AntMaze,
    # `terminals` is a success indicator and can be true for many consecutive
    # goal transitions. Splitting on it creates thousands of 1-step fragments and
    # destroys positive preference segments. Discontinuities recover 999 x 1000
    # step trajectories for antmaze-medium-play-v2.
    ends=None
    if mode in ("auto", "discontinuity") and "observations" in data and "next_observations" in data:
        obs=np.asarray(data["observations"]); nxt=np.asarray(data["next_observations"])
        if len(obs)>1 and len(nxt)==len(obs):
            diff=np.max(np.abs(nxt[:-1]-obs[1:]), axis=1)
            disc=np.where(diff>1e-5)[0]
            if mode=="discontinuity" or len(disc)>0:
                ends=disc
    if ends is None:
        if mode=="timeouts":
            ends=np.where(timeouts>0.5)[0]
        elif mode=="terminals":
            ends=np.where((terminals>0.5)|(timeouts>0.5))[0]
        else:
            # Auto fallback for non-AntMaze datasets without row discontinuities.
            ends=np.where((terminals>0.5)|(timeouts>0.5))[0]

    out=[]; s=0
    for e in ends:
        if e+1>s: out.append((s,e+1))
        s=e+1
    if s<len(terminals): out.append((s,len(terminals)))
    return out

def episode_start_index(data):
    ep_start=np.zeros(len(data["rewards"]), dtype=np.int64)
    for s,e in episode_slices(data): ep_start[s:e]=s
    return ep_start

def summarize(data):
    print("N", len(data["rewards"]), "obs", data["observations"].shape, "act", data["actions"].shape, flush=True)
    print("reward mean/min/max/pos", float(data["rewards"].mean()), float(data["rewards"].min()), float(data["rewards"].max()), int((data["rewards"]>0).sum()), flush=True)
    print("episodes", len(episode_slices(data)), flush=True)

# ---- Preference dataset ----
def build_segment_start_pool(data, H: int):
    rewards=np.asarray(data["rewards"], dtype=np.float32)
    starts=[]; rets=[]
    cs=np.concatenate([[0.0], np.cumsum(rewards, dtype=np.float64)])
    for s,e in episode_slices(data):
        if e-s < H: continue
        st=np.arange(s, e-H+1, dtype=np.int64)
        rt=(cs[st+H]-cs[st]).astype(np.float32)
        starts.append(st); rets.append(rt)
    starts=np.concatenate(starts); rets=np.concatenate(rets)
    pos=np.where(rets>0)[0]; zero=np.where(rets<=0)[0]
    print("segment starts", len(starts), "positive segments", len(pos), flush=True)
    return starts, rets, pos, zero

def build_synthetic_preference_pairs(data, budget:int, H:int, seed:int):
    rng=np.random.default_rng(seed)
    starts, seg_ret, pos, zero = build_segment_start_pool(data,H)
    a=np.empty(budget,dtype=np.int64); b=np.empty(budget,dtype=np.int64); y=np.empty(budget,dtype=np.float32)
    informative=0
    for i in range(budget):
        use_inf = len(pos)>0 and len(zero)>0 and rng.random() < CONFIG["informative_pair_frac"]
        if use_inf:
            ip=pos[rng.integers(len(pos))]; iz=zero[rng.integers(len(zero))]
            if rng.random()<0.5:
                ia,ib=ip,iz; label=1.0
            else:
                ia,ib=iz,ip; label=0.0
            informative += 1
        else:
            ia,ib=rng.integers(len(starts), size=2)
            ra,rb=seg_ret[ia], seg_ret[ib]
            label=1.0 if ra>rb else 0.0 if rb>ra else 0.5
        a[i]=starts[ia]; b[i]=starts[ib]; y[i]=label
    return a,b,y,{"informative_pairs": informative, "tie_frac": float(np.mean(y==0.5)), "label_mean": float(y.mean())}

def _first_existing(base: Path, prefixes):
    files=sorted([p for p in base.iterdir() if p.is_file()])
    for pref in prefixes:
        matched=[p for p in files if p.name.startswith(pref)]
        if matched: return matched[0]
    raise FileNotFoundError(f"missing any of {prefixes} under {base}")

def load_human_label_arrays(env_name: str):
    base=Path(CONFIG["human_label_dir"]) / env_name
    if not base.exists():
        raise FileNotFoundError(f"human label directory not found: {base}")
    i1_path=_first_existing(base, ["indices_num"])
    i2_path=_first_existing(base, ["indices_2_num"])
    label_path=base / "label_human"
    with i1_path.open('rb') as f: i1=np.asarray(pickle.load(f), dtype=np.int64)
    with i2_path.open('rb') as f: i2=np.asarray(pickle.load(f), dtype=np.int64)
    with label_path.open('rb') as f: labels=np.asarray(pickle.load(f), dtype=np.int64)
    n=min(len(i1), len(i2), len(labels))
    return i1[:n], i2[:n], labels[:n], {"i1_file": i1_path.name, "i2_file": i2_path.name, "label_file": label_path.name}

def build_human_preference_pairs(data, budget:int, H:int, seed:int):
    i1,i2,labels,files=load_human_label_arrays(CONFIG["env_name"])
    total=len(labels)
    if budget > total:
        raise ValueError(f"budget {budget} exceeds available human labels {total}")
    mx=int(max(i1.max(), i2.max())) if total else -1
    if mx + H > len(data["rewards"]):
        raise ValueError(f"human label segment end {mx+H} exceeds dataset length {len(data['rewards'])}; check H/query_len")
    mode=CONFIG["human_budget_split"]
    if mode == "tail":
        subset=np.arange(total-budget, total, dtype=np.int64)
    elif mode == "prefix":
        subset=np.arange(budget, dtype=np.int64)
    elif mode == "nested_shuffle":
        subset=np.random.default_rng(CONFIG["data_seed"]).permutation(total)[:budget]
    else:
        raise ValueError(f"unknown HUMAN_BUDGET_SPLIT={mode}")
    lab=labels[subset]
    # Official PreferenceTransformer convention: 0 => first segment, 1 => second, -1 => tie.
    y=np.where(lab==0, 1.0, np.where(lab==1, 0.0, 0.5)).astype(np.float32)
    stats={
        "label_source":"human",
        "human_total_labels": int(total),
        "human_budget_split": mode,
        "human_query_len": int(H),
        "human_label_0_first": int(np.sum(lab==0)),
        "human_label_1_second": int(np.sum(lab==1)),
        "human_label_neg1_tie": int(np.sum(lab==-1)),
        "tie_frac": float(np.mean(y==0.5)),
        "label_mean": float(y.mean()),
        **files,
    }
    return i1[subset].astype(np.int64), i2[subset].astype(np.int64), y, stats

def build_preference_pairs(data, budget:int, H:int, seed:int):
    if CONFIG["label_source"] == "human":
        return build_human_preference_pairs(data, budget, H, seed)
    if CONFIG["label_source"] == "synthetic":
        return build_synthetic_preference_pairs(data, budget, H, seed)
    raise ValueError(f"unknown LABEL_SOURCE={CONFIG['label_source']}")

# ---- Reward models ----
class MRReward(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__(); self.net=mlp([obs_dim+act_dim, *CONFIG["hidden_dims"], 1])
    def step_reward(self, obs, act): return self.net(torch.cat([obs,act], -1)).squeeze(-1)
    def segment_score(self, obs_seq, act_seq):
        B,H,D=obs_seq.shape; r=self.step_reward(obs_seq.reshape(B*H,D), act_seq.reshape(B*H,-1)).reshape(B,H)
        return r.sum(1)
    def final_reward(self, obs_seq, act_seq): return self.step_reward(obs_seq[:,-1], act_seq[:,-1])

class NMRReward(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__(); h=CONFIG["seq_hidden"]; self.inp=nn.Linear(obs_dim+act_dim,h); self.lstm=nn.LSTM(h,h,batch_first=True); self.head=nn.Linear(h,1)
    def rewards(self, obs_seq, act_seq):
        x=torch.cat([obs_seq,act_seq], -1); h=torch.relu(self.inp(x)); out,_=self.lstm(h); return self.head(out).squeeze(-1)
    def segment_score(self, obs_seq, act_seq): return self.rewards(obs_seq,act_seq).sum(1)
    def final_reward(self, obs_seq, act_seq): return self.rewards(obs_seq,act_seq)[:,-1]

class PTReward(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__(); d=CONFIG["seq_hidden"]; H=CONFIG["segment_len"]
        self.inp=nn.Linear(obs_dim+act_dim,d); self.pos=nn.Parameter(torch.zeros(1,H,d))
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=CONFIG["pt_heads"],dim_feedforward=4*d,dropout=0.1,batch_first=True,activation="gelu")
        self.causal=nn.TransformerEncoder(enc,num_layers=CONFIG["pt_layers"])
        att=nn.TransformerEncoderLayer(d_model=d,nhead=CONFIG["pt_heads"],dim_feedforward=4*d,dropout=0.1,batch_first=True,activation="gelu")
        self.bidir=nn.TransformerEncoder(att,num_layers=1)
        self.r=nn.Linear(d,1); self.w=nn.Linear(d,1)
        mask=torch.triu(torch.ones(H,H), diagonal=1).bool(); self.register_buffer("causal_mask", mask)
    def outputs(self, obs_seq, act_seq):
        B,H,_=obs_seq.shape; x=torch.cat([obs_seq,act_seq], -1); h=self.inp(x)+self.pos[:,:H]
        h=self.causal(h, mask=self.causal_mask[:H,:H]); z=self.bidir(h)
        r=self.r(z).squeeze(-1); wlog=self.w(z).squeeze(-1); return r,wlog
    def segment_score(self, obs_seq, act_seq):
        r,wlog=self.outputs(obs_seq,act_seq); w=torch.softmax(wlog, dim=1); return (w*r).sum(1)
    def final_reward(self, obs_seq, act_seq): return self.outputs(obs_seq,act_seq)[0][:,-1]

def make_reward_model(kind, obs_dim, act_dim):
    return {"mr":MRReward,"nmr":NMRReward,"pt":PTReward}[kind](obs_dim,act_dim).to(device)

# ---- Reward training / relabel ----
def normalize_obs(data):
    obs=data["observations"].astype(np.float32); mean=obs.mean(0,keepdims=True); std=obs.std(0,keepdims=True)+1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def seq_batch(obs_np, act_np, starts, H):
    idx=starts[:,None]+np.arange(H,dtype=np.int64)[None,:]
    return torch.as_tensor(obs_np[idx], device=device), torch.as_tensor(act_np[idx], device=device)

def train_reward_model(kind, data, budget, seed):
    seed_everything(seed)
    H=CONFIG["segment_len"]; obs_dim=data["observations"].shape[1]; act_dim=data["actions"].shape[1]
    obs_mean, obs_std=normalize_obs(data); obs_np=((data["observations"]-obs_mean)/obs_std).astype(np.float32); act_np=data["actions"].astype(np.float32)
    a,b,y,stats=build_preference_pairs(data,budget,H,seed)
    y_t=torch.as_tensor(y, device=device)
    net=make_reward_model(kind,obs_dim,act_dim); opt=torch.optim.AdamW(net.parameters(), lr=CONFIG["reward_lr"], weight_decay=1e-4)
    B=min(CONFIG["pref_batch_size"], budget); t0=time.time(); logs=[]
    for step in range(1, CONFIG["reward_steps"]+1):
        j=torch.randint(0,budget,(B,),device=device).cpu().numpy()
        oa,aa=seq_batch(obs_np,act_np,a[j],H); ob,ab=seq_batch(obs_np,act_np,b[j],H)
        logits=net.segment_score(oa,aa)-net.segment_score(ob,ab)
        loss=F.binary_cross_entropy_with_logits(logits, y_t[j])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if step==1 or step%CONFIG["log_every"]==0 or step==CONFIG["reward_steps"]:
            with torch.no_grad():
                pred=(torch.sigmoid(logits)>0.5).float(); hard=(y_t[j]>0.5).float(); mask=(y_t[j]!=0.5)
                acc=float((pred[mask]==hard[mask]).float().mean().item()) if mask.any() else None
            row={"phase":"reward","model":kind,"budget":budget,"step":step,"loss":float(loss.item()),"batch_acc":acc,"elapsed_sec":time.time()-t0}
            print(row, flush=True); logs.append(row)
    # full-pair train accuracy in chunks
    net.eval(); correct=0; total=0
    with torch.no_grad():
        for s in range(0,budget,512):
            sl=slice(s,min(s+512,budget)); oa,aa=seq_batch(obs_np,act_np,a[sl],H); ob,ab=seq_batch(obs_np,act_np,b[sl],H)
            yy=torch.as_tensor(y[sl],device=device); mask=(yy!=0.5)
            if mask.any():
                pred=(torch.sigmoid(net.segment_score(oa,aa)-net.segment_score(ob,ab))>0.5).float(); hard=(yy>0.5).float()
                correct += int((pred[mask]==hard[mask]).sum().item()); total += int(mask.sum().item())
    stats["pref_train_acc"] = correct/total if total else None
    return net, obs_mean, obs_std, stats, logs

def context_indices(ep_start, start, end, H):
    i=np.arange(start,end,dtype=np.int64); s=np.maximum(ep_start[i], i-H+1); lengths=i-s+1
    pos=np.arange(H,dtype=np.int64)[None,:]; rel=pos-(H-lengths[:,None]); return np.where(rel<0, s[:,None], s[:,None]+rel)

def relabel_rewards(kind, net, data, obs_mean, obs_std):
    obs_np=((data["observations"]-obs_mean)/obs_std).astype(np.float32); act_np=data["actions"].astype(np.float32)
    ep_start=episode_start_index(data); H=CONFIG["segment_len"]; out=[]
    net.eval()
    with torch.no_grad():
        if kind=="mr":
            for s in range(0,len(obs_np),8192):
                o=torch.as_tensor(obs_np[s:s+8192],device=device); a=torch.as_tensor(act_np[s:s+8192],device=device)
                out.append(net.step_reward(o,a).cpu().numpy())
        else:
            for s in range(0,len(obs_np),2048):
                idx=context_indices(ep_start,s,min(s+2048,len(obs_np)),H)
                o=torch.as_tensor(obs_np[idx],device=device); a=torch.as_tensor(act_np[idx],device=device)
                out.append(net.final_reward(o,a).cpu().numpy())
    raw=np.concatenate(out).astype(np.float32)
    lo,hi=np.percentile(raw,[5,95])
    cal=np.zeros_like(raw) if hi-lo<1e-6 else np.clip((raw-lo)/(hi-lo),0,1).astype(np.float32)
    return raw, cal

# ---- IQL ----
class ReplayBuffer:
    def __init__(self,data,reward_key):
        obs=data["observations"].astype(np.float32); next_obs=data["next_observations"].astype(np.float32)
        self.obs_mean=obs.mean(0,keepdims=True); self.obs_std=obs.std(0,keepdims=True)+1e-6
        self.obs=torch.as_tensor((obs-self.obs_mean)/self.obs_std,device=device)
        self.next_obs=torch.as_tensor((next_obs-self.obs_mean)/self.obs_std,device=device)
        self.actions=torch.as_tensor(data["actions"].astype(np.float32),device=device)
        self.rewards=torch.as_tensor(data[reward_key].astype(np.float32).reshape(-1,1),device=device)
        self.not_dones=torch.as_tensor((1.0-data["terminals"].astype(np.float32)).reshape(-1,1),device=device)
        self.size=len(obs)
    def sample(self,B):
        idx=torch.randint(0,self.size,(B,),device=device)
        return {"obs":self.obs[idx],"actions":self.actions[idx],"rewards":self.rewards[idx],"next_obs":self.next_obs[idx],"not_dones":self.not_dones[idx]}
class TwinQ(nn.Module):
    def __init__(self,od,ad): super().__init__(); self.q1=mlp([od+ad,*CONFIG["hidden_dims"],1]); self.q2=mlp([od+ad,*CONFIG["hidden_dims"],1])
    def both(self,o,a): x=torch.cat([o,a],-1); return self.q1(x),self.q2(x)
    def forward(self,o,a): q1,q2=self.both(o,a); return torch.minimum(q1,q2)
class ValueNet(nn.Module):
    def __init__(self,od): super().__init__(); self.v=mlp([od,*CONFIG["hidden_dims"],1])
    def forward(self,o): return self.v(o)
class GaussianPolicy(nn.Module):
    def __init__(self,od,ad): super().__init__(); h=CONFIG["hidden_dims"][-1]; self.backbone=mlp([od,*CONFIG["hidden_dims"],h]); self.mean=nn.Linear(h,ad); self.log_std=nn.Linear(h,ad)
    def forward(self,o): h=self.backbone(o); return self.mean(h), self.log_std(h).clamp(-5,2)
    def log_prob(self,o,a):
        eps=1e-6; a=a.clamp(-1+eps,1-eps); pre=0.5*(torch.log1p(a)-torch.log1p(-a)); m,ls=self(o); dist=Normal(m,ls.exp())
        return (dist.log_prob(pre)-torch.log(1-a.pow(2)+eps)).sum(-1,keepdim=True)
    @torch.no_grad()
    def act(self,o):
        if o.ndim==1: o=o[None]
        m,_=self(o); return torch.tanh(m).cpu().numpy()[0]
def expectile_loss(diff):
    w=torch.where(diff>0, CONFIG["expectile"], 1-CONFIG["expectile"]); return (w*diff.pow(2)).mean()
class IQLAgent:
    def __init__(self,od,ad):
        self.q=TwinQ(od,ad).to(device); self.q_target=TwinQ(od,ad).to(device); self.q_target.load_state_dict(self.q.state_dict())
        self.v=ValueNet(od).to(device); self.policy=GaussianPolicy(od,ad).to(device)
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=CONFIG["lr"]); self.v_opt=torch.optim.Adam(self.v.parameters(),lr=CONFIG["lr"]); self.pi_opt=torch.optim.Adam(self.policy.parameters(),lr=CONFIG["lr"])
    def update(self,b):
        with torch.no_grad(): tq= b["rewards"] + CONFIG["discount"]*b["not_dones"]*self.v(b["next_obs"])
        q1,q2=self.q.both(b["obs"],b["actions"]); q_loss=F.mse_loss(q1,tq)+F.mse_loss(q2,tq)
        self.q_opt.zero_grad(set_to_none=True); q_loss.backward(); self.q_opt.step()
        with torch.no_grad(): q_min=self.q_target(b["obs"],b["actions"])
        v=self.v(b["obs"]); v_loss=expectile_loss(q_min-v)
        self.v_opt.zero_grad(set_to_none=True); v_loss.backward(); self.v_opt.step()
        with torch.no_grad(): adv=q_min-v; w=torch.exp(CONFIG["temperature"]*adv).clamp(max=CONFIG["max_adv_weight"])
        pi_loss=-(w*self.policy.log_prob(b["obs"],b["actions"])).mean()
        self.pi_opt.zero_grad(set_to_none=True); pi_loss.backward(); self.pi_opt.step()
        with torch.no_grad():
            for p,tp in zip(self.q.parameters(),self.q_target.parameters()): tp.data.mul_(1-CONFIG["tau"]).add_(CONFIG["tau"]*p.data)
        return {"v_loss":float(v_loss.item()),"q_loss":float(q_loss.item()),"pi_loss":float(pi_loss.item()),"adv_mean":float(adv.mean().item()),"w_mean":float(w.mean().item())}

def gym_reset(env):
    out=env.reset(); return out[0] if isinstance(out,tuple) else out
def gym_step(env,a):
    out=env.step(a)
    if len(out)==5:
        o,r,term,trunc,info=out; return o,r,term or trunc,info
    return out

def evaluate(env,agent,replay):
    returns=[]
    for _ in range(CONFIG["eval_episodes"]):
        obs=gym_reset(env); ep=0.0
        for _ in range(1000):
            on=((np.asarray(obs,dtype=np.float32)[None]-replay.obs_mean)/replay.obs_std).astype(np.float32)
            act=agent.policy.act(torch.as_tensor(on,device=device)); obs,r,d,info=gym_step(env,act); ep+=float(r)
            if d: break
        returns.append(ep)
    avg=float(np.mean(returns)) if returns else None
    norm=None
    if avg is not None:
        try: norm=float(env.get_normalized_score(avg)*100.0)
        except Exception: norm=None
    return {"eval_return":avg,"normalized_score":norm}

def train_iql(data, env, reward_key, label):
    seed_everything(CONFIG["seed"] + abs(hash(label))%100000)
    replay=ReplayBuffer(data,reward_key); agent=IQLAgent(data["observations"].shape[1], data["actions"].shape[1])
    logs=[]; t0=time.time()
    for step in trange(1, CONFIG["iql_steps"]+1, desc=f"IQL {label}"):
        metrics=agent.update(replay.sample(CONFIG["batch_size"]))
        if step==1 or step%CONFIG["log_every"]==0 or step==CONFIG["iql_steps"]:
            row={"phase":"iql","label":label,"step":step,"elapsed_sec":time.time()-t0,**metrics}; print(row, flush=True); logs.append(row)
    return agent,replay,logs,evaluate(env,agent,replay)


def save_agent_checkpoint(agent, replay, label, out_dir):
    path=Path(out_dir)/f"{label}_iql_policy.pt"
    torch.save({
        "label": label,
        "config": CONFIG,
        "policy": agent.policy.state_dict(),
        "q": agent.q.state_dict(),
        "q_target": agent.q_target.state_dict(),
        "v": agent.v.state_dict(),
        "obs_mean": replay.obs_mean.astype(np.float32),
        "obs_std": replay.obs_std.astype(np.float32),
    }, path)
    print("CHECKPOINT", path, flush=True)
    return str(path)

def antmaze_train_rewards(r): return r.astype(np.float32)-1.0


def _base_env(env):
    return getattr(env, 'unwrapped', env)

def _rowcol_to_xy_for_plot(base, row, col):
    scale=getattr(base, '_maze_size_scaling', 4.0)
    init_x=getattr(base, '_init_torso_x', scale)
    init_y=getattr(base, '_init_torso_y', scale)
    return col*scale-init_x, row*scale-init_y

def plot_policy_rollouts(env, agent, replay, label, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    base=_base_env(env)
    maze=getattr(base, '_maze_map', None)
    if maze is None:
        print('rollout plot skipped: env has no _maze_map', flush=True); return []
    scale=getattr(base, '_maze_size_scaling', 4.0)
    paths=[]
    for ep in range(CONFIG['rollout_episodes']):
        obs=gym_reset(env)
        target=np.asarray(getattr(base, 'target_goal', [np.nan, np.nan]), dtype=float)
        xy=[]; rewards=[]
        for t in range(CONFIG['rollout_max_steps']):
            cur_xy=np.asarray(base.get_xy() if hasattr(base, 'get_xy') else np.asarray(obs)[:2], dtype=float)
            xy.append(cur_xy.copy())
            on=((np.asarray(obs,dtype=np.float32)[None]-replay.obs_mean)/replay.obs_std).astype(np.float32)
            act=agent.policy.act(torch.as_tensor(on,device=device))
            obs,r,d,info=gym_step(env,act); rewards.append(float(r))
            if d: break
        if len(xy)==0: continue
        xy=np.asarray(xy)
        ret=float(np.sum(rewards)); success=int(np.sum(rewards)>0)
        fig,ax=plt.subplots(figsize=(7.0,7.8), dpi=180)
        # Maze cells.
        for i,row in enumerate(maze):
            for j,cell in enumerate(row):
                cx,cy=_rowcol_to_xy_for_plot(base,i,j)
                is_wall=(cell==1)
                color='#114c57' if is_wall else '#fff2a6'
                rect=Rectangle((cx-scale/2, cy-scale/2), scale, scale, facecolor=color,
                               edgecolor='#12363d', linewidth=1.0, alpha=0.98)
                ax.add_patch(rect)
                if cell == 'r':
                    ax.scatter([cx],[cy], s=120, c='#35c9bd', edgecolors='#14343a', linewidths=1.5,
                               label='maze reset', zorder=5)
                if cell == 'g':
                    ax.scatter([cx],[cy], s=180, marker='D', facecolors='none', edgecolors='#7b2cbf', linewidths=2.0,
                               label='maze goal cell', zorder=5)
        # Rollout path.
        ax.plot(xy[:,0], xy[:,1], color='#ff7f0e', linewidth=2.0, label=f'{label} rollout', zorder=6)
        ax.scatter([xy[0,0]],[xy[0,1]], s=110, c='#39d47a', edgecolors='black', linewidths=1.2, label='start', zorder=7)
        ax.scatter([xy[-1,0]],[xy[-1,1]], s=110, c='#ff3b6b', marker='X', edgecolors='black', linewidths=1.0, label='end', zorder=7)
        if np.isfinite(target).all():
            ax.scatter([target[0]],[target[1]], s=180, c='#ffd166', marker='*', edgecolors='black', linewidths=1.0,
                       label='target goal', zorder=8)
        ax.set_aspect('equal')
        ax.set_xlim(-6, 26); ax.set_ylim(26, -6)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'antmaze-medium-play-v2 {label} rollout | return={ret:.1f}, success={success}')
        ax.grid(True, color='#183d44', linewidth=0.8, alpha=0.55)
        handles, labels = ax.get_legend_handles_labels()
        seen=set(); uh=[]; ul=[]
        for h,l in zip(handles,labels):
            if l not in seen:
                uh.append(h); ul.append(l); seen.add(l)
        ax.legend(uh, ul, loc='upper center', bbox_to_anchor=(0.5,-0.08), ncol=3, frameon=True)
        fig.tight_layout()
        out=Path(out_dir)/f'{label}_rollout_ep{ep}.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(out))
        np.savez(Path(out_dir)/f'{label}_rollout_ep{ep}.npz', xy=xy, rewards=np.asarray(rewards), target=target, ret=ret, success=success)
        print('ROLLOUT_PLOT', out, 'return', ret, 'success', success, flush=True)
    return paths

# ---- Main ----
def append_csv(path, row):
    exists=path.exists(); fields=sorted(row.keys())
    if exists:
        with path.open() as f: fields=f.readline().strip().split(',')
    with path.open('a', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); 
        if not exists: w.writeheader()
        w.writerow(row)

results_path=RUN_DIR/"results.csv"; logs_path=RUN_DIR/"train_logs.jsonl"
with (RUN_DIR/"config.json").open('w') as f: json.dump(CONFIG,f,indent=2)

data, env=load_d4rl(CONFIG["env_name"]); summarize(data)
all_logs=[]
if CONFIG["run_oracle"]:
    data["iql_oracle"]=antmaze_train_rewards(data["rewards"])
    _,_,logs,ev=train_iql(data,env,"iql_oracle","oracle")
    row={"setting":"oracle","model":"oracle","budget":"",**ev,**logs[-1]}; append_csv(results_path,row); all_logs+=logs

for budget in CONFIG["budgets"]:
    for kind in CONFIG["models"]:
        label=f"{kind}_budget{budget}"
        print("\n===", label, "===", flush=True)
        net,om,osd,pref_stats,rlogs=train_reward_model(kind,data,budget,CONFIG["seed"]+budget+{"mr":11,"nmr":22,"pt":33}[kind])
        raw,cal=relabel_rewards(kind,net,data,om,osd)
        rew=cal.astype(np.float32)
        if CONFIG["antmaze_reward_shift"] and "antmaze" in CONFIG["env_name"]: rew=rew-1.0
        key="iql_"+label; data[key]=rew
        agent,replay,ilogs,ev=train_iql(data,env,key,label)
        row={"setting":label,"model":kind,"budget":budget,**ev,**pref_stats,
             "pred_raw_mean":float(raw.mean()),"pred_raw_std":float(raw.std()),"pred_cal_mean":float(cal.mean()),"pred_cal_std":float(cal.std()),
             "final_v_loss":ilogs[-1]["v_loss"],"final_q_loss":ilogs[-1]["q_loss"],"final_pi_loss":ilogs[-1]["pi_loss"]}
        print("RESULT", row, flush=True); append_csv(results_path,row)
        if CONFIG.get("save_checkpoints", False):
            try:
                save_agent_checkpoint(agent, replay, label, RUN_DIR)
            except Exception as e:
                print("checkpoint save failed", repr(e), flush=True)
        if CONFIG.get("save_rollouts", False):
            try:
                plot_policy_rollouts(env, agent, replay, label, RUN_DIR)
            except Exception as e:
                print("rollout plot failed", repr(e), flush=True)
        all_logs+=rlogs+ilogs
        with logs_path.open('a') as f:
            for x in rlogs+ilogs: f.write(json.dumps(x)+"\n")

# summary/plot
try:
    import pandas as pd, matplotlib.pyplot as plt
    df=pd.read_csv(results_path); print("\nFINAL RESULTS\n", df, flush=True)
    try:
        run_dir_display = str(RUN_DIR.relative_to(PROJECT_DIR))
    except ValueError:
        run_dir_display = str(RUN_DIR)
    summary_lines=["# Label budget comparison", "", "Run dir: `"+run_dir_display+"`", ""]
    summary_lines.append(df.to_csv(index=False))
    (RUN_DIR/"RESULT_SUMMARY.md").write_text("\n".join(summary_lines))
    plt.figure(figsize=(7,4))
    for kind,sub in df[df['model']!='oracle'].groupby('model'):
        sub=sub.sort_values('budget'); plt.plot(sub['budget'], sub['normalized_score'], marker='o', label=kind.upper())
    plt.xscale('log'); plt.xlabel('Preference label budget'); plt.ylabel('D4RL normalized score'); plt.title(CONFIG['env_name']); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR/"model_budget_comparison.png", dpi=160)
except Exception as e:
    print('summary plot failed', repr(e), flush=True)
print("DONE", RUN_DIR, flush=True)
