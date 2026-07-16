import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════


@app.cell
def _():
    import marimo as mo
    import json
    import sys
    import torch
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from pathlib import Path
    from sklearn.decomposition import PCA
    from huggingface_hub import hf_hub_download

    _MODEL_DIR = Path(__file__).parent / "model"
    _MODEL_DIR.mkdir(exist_ok=True)
    for _f in ["layers.py", "trm.py"]:
        if not (_MODEL_DIR / _f).exists():
            hf_hub_download(
                repo_id="davp-sapienza/trm-checkpoints",
                filename=f"model/{_f}",
                local_dir=Path(__file__).parent,
            )

    sys.path.insert(0, str(Path(__file__).parent))
    from model.trm import TinyRecursiveModel
    from model.layers import TransformerBlock

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DEVICE, PCA, Path, TransformerBlock, TinyRecursiveModel,
        go, hf_hub_download, json, mo, np, px, sys, torch,
    )


# ═══════════════════════════════════════════════════════════════
#  TITLE & ABSTRACT
# ═══════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    GITHUB_REPO = "https://github.com/the-summoning/alphaXiv-marimo"

    mo.md(
        f"""
        # Latent Dynamics and Efficient Inference in Tiny Recursive Models

        **Davide Perniconi** — *MSc Student at Sapienza University of Rome*

        ---

        ## Abstract

        Tiny Recursive Model (TRM) is a recent recursive architecture that outperforms frontier LLMs on reasoning tasks.
        Despite being called "tiny", TRMs require many sequential steps to produce accurate predictions on non-trivial problems.
        Furthermore, a recent extension trades off even more inference time for accuracy by requiring dozens of parallel rollouts.
        At the same time, the internal evolution of TRM hidden states remains largely unexplored.
        This work is twofold: first, it characterizes the geometry of the latent reasoning space of TRMs; second, it addresses the inference-time bottleneck by applying an acceleration-based early stopping criterion and evaluating its impact.

        ---

        *Interactive notebook — data from [HuggingFace](https://huggingface.co/davp-sapienza). | Code: [GitHub]({GITHUB_REPO})*
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
#  §1  INTRODUCTION & RELATED WORK
# ═══════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(
        r"""
        ## §1  Related Work

        The work is grounded in recent advancements in latent reasoning architectures:

        1. **Tiny Recursive Model (TRM):** [Jolicoeur-Martineau (2025)](https://arxiv.org/abs/2510.04871) simplifies the biology-inspired HRM architecture ([Wang et al., 2025](https://arxiv.org/abs/2506.21734)) by iteratively processing hidden states through a single 2-layer Transformer with deep supervision.

        2. **Geometric metrics and early stopping:** [Pappone et al. (2025)](https://arxiv.org/abs/2509.23314) show that internal representations within recurrent Transformer blocks evolve through a combination of fine-grained local refinements and larger-scale drift, and introduces geometric metrics to compare different early-exit mechanisms.

        3. **Probabilistic TRM (PTRM):** [Sghaier, Parviz, Jolicoeur-Martineau (2026)](https://arxiv.org/abs/2605.19943) propose a simple training-free extension that blindly injects noise during recursion. Although PTRM achieves substantial accuracy gains, it significantly increases inference costs by requiring up to 100 parallel rollouts per sample.

        The **main contributions** of this work are:

        1. implementation from scratch of the TRM (both MLP-Mixer and Attention variants)
        2. study of both reasoning and solution latents ($y$ and $z$)
        3. implementation of acceleration-based early stopping to reduce inference time, and
        4. experimental investigation of Matryoshka Representation Learning imposed on $y$.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
#  §2  METHOD
# ═══════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(
        r"""
        ## §2  Method

        TRMs were trained both with and without **Matryoshka Representation Learning (MRL)** ([Kusupati et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)).
        For the geometric analysis, the latent trajectories were studied via **PCA** for visualization, and the metrics introduced by [Pappone et al. (2025)](https://arxiv.org/abs/2509.23314) were computed.
        The trained models were evaluated and the resulting latent trajectories were recorded for direct inspection of the internal computation process. Additional trajectories were collected from the publicly available [ARC-AGI checkpoints](https://huggingface.co/arcprize/trm_arc_prize_verification).
        Finally, TRM was extended with the early stopping criterion based on the **unnormalized acceleration** (see [Equation 2](#eq-acceleration)) with two-hit check [Pappone et al. (2025)](https://arxiv.org/abs/2509.23314).
        The objective is to achieve approximately the same accuracy obtained using all supervision steps ($N_{\text{sup}} = 16$) while reducing the average number of iterations during inference.

        ### Formulas

        The $\Delta$ factor used for acceleration is computed on solution latents $y$ and defined as ($k$ denotes supervision step):

        <a id="eq-delta"></a>
        $$\Delta^{(k)} = y^{(k+1)} - y^{(k)} \qquad \textbf{(1)}$$

        The standard unnormalized acceleration is defined as:

        <a id="eq-acceleration"></a>
        $$a^{(k)} = \big\lVert \Delta^{(k)} - \Delta^{(k-1)} \big\rVert_2 \qquad \textbf{(2)}$$

        The normalized acceleration (drop-in replacement) is defined as:

        <a id="eq-norm-acceleration"></a>
        $$\hat{a}^{(k)} = \frac{\left\|\Delta^{(k)} - \Delta^{(k-1)}\right\|_{2}}{\left\|\Delta^{(k)}\right\|_{2} + \left\|\Delta^{(k-1)}\right\|_{2} + \varepsilon} \in [0,2] \qquad \textbf{(3)}$$
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
#  §3  EXPERIMENTAL RESULTS
# ═══════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(
        r"""
        ## §3  Experimental Results

        <a id="s3-setup"></a>
        ### Setup

        We train TRMs on Maze 5x5, Maze 10x10 (generated using the [`maze-dataset` library](https://arxiv.org/abs/2309.10498)), and Sudoku 9x9 (10k samples with difficulty $\leq 2$ sourced from [Hugging Face](https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset)).
        For all tasks, the internal recursive block of the TRM is configured as a compact 2-layer Transformer with
        $N_{\text{sup}} = 16$ supervision steps, $T = 4$ deep recursion steps, and $n = 8$ latent recursion steps. All high-dimensional latent trajectories are analyzed via Principal Component Analysis (PCA).

        | Task | Hidden | Heads | $n$ | $T$ | Registers | Mixing | MRL from |
        |------|--------|-------|-----|-----|-----------|--------|----------|
        | Maze 5×5 | 16 | 2 | 8 | 4 | 3 | MLP-T | — |
        | Maze 10×10 | 32 | 2 | 8 | 4 | 3 | MLP-T | — |
        | Sudoku 9×9 | 16 | 2 | 8 | 4 | 4 | MLP-T | — |
        | Sudoku 9×9 (attn) | 128 | 2 | 8 | 4 | 4 | Attention | — |
        | Maze 5×5 (MRL) | 16 | 2 | 8 | 4 | 3 | MLP-T | dim 1 |
        | Maze 10×10 (MRL) | 32 | 2 | 8 | 4 | 3 | MLP-T | dim 1 |
        | Sudoku 9×9 (MRL) | 16 | 2 | 8 | 4 | 4 | MLP-T | dim 4 |
        | Sudoku 9×9 (MRL attn) | 128 | 2 | 8 | 4 | 4 | Attention | dim 16 |
        """
    )
    return


@app.cell
def _():
    DATASET_FILES = {
        "Maze 5×5 (no MRL)": "maze5x5_noMRL.npz",
        "Maze 5×5 (MRL)": "maze5x5_withMRL.npz",
        "Maze 10×10 (no MRL)": "maze10x10_noMRL.npz",
        "Maze 10×10 (MRL)": "maze10x10_withMRL.npz",
        "Sudoku 9×9 (no MRL)": "sudoku9x9_noMRL.npz",
        "Sudoku 9×9 (attn)": "sudoku9x9_noMRL_attn.npz",
        "Sudoku 9×9 (MRL)": "sudoku9x9_withMRL.npz",
        "Sudoku 9×9 (MRL, attn)": "sudoku9x9_withMRL_attention.npz",
        "ARC-AGI v1 (test)": "agi_test_traj.npz",
        "ARC-AGI v2 (test)": "agi2_test_traj.npz",
    }
    NON_AGI_FILES = {k: v for k, v in DATASET_FILES.items() if "ARC-AGI" not in k}
    return DATASET_FILES, NON_AGI_FILES


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Results without early stopping

        Below are reported the results obtained when using different supervision steps ($N_{\text{sup}}$). Maze tasks appear to be
        solvable with just one supervision step while Sudoku requires at least $N_{\text{sup}} = 2$.

        <img src="https://huggingface.co/datasets/davp-sapienza/trm-datasets/resolve/main/results.png"
             alt="Results tables" style="width:100%; max-width:900px; display:block; margin:0 auto;">
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        *Try selecting a task and varying the number of supervision steps below to see the model solve puzzles in real time.*
        """
    )
    return


@app.cell
def _(mo):
    task_dropdown_s1 = mo.ui.dropdown(
        options=[
            "Maze 5x5", "Maze 5x5 (MRL)",
            "Maze 10x10", "Maze 10x10 (MRL)",
            "Sudoku 9x9", "Sudoku 9x9 (MRL)",
            "Sudoku 9x9 (attention)", "Sudoku 9x9 (attention, MRL)",
        ],
        value="Maze 5x5",
        label="Task",
    )
    step_slider_s1 = mo.ui.slider(
        start=1, stop=16, step=1, value=16,
        label=r"Supervision Steps  $N_{\text{sup}}$",
        show_value=True,
    )
    mo.hstack([task_dropdown_s1, step_slider_s1], gap=2)
    return step_slider_s1, task_dropdown_s1


@app.cell
def _(hf_hub_download, mo, np, task_dropdown_s1):
    _TASK_MAP = {
        "Maze 5x5": ("maze5x5_noMRL", 81, "maze/maze5x5/test"),
        "Maze 5x5 (MRL)": ("maze5x5_MRL", 81, "maze/maze5x5/test"),
        "Maze 10x10": ("maze10x10_noMRL", 361, "maze/maze10x10/test"),
        "Maze 10x10 (MRL)": ("maze10x10_MRL", 361, "maze/maze10x10/test"),
        "Sudoku 9x9": ("sudoku-9x9_noMRL", 81, "sudoku9x9/val"),
        "Sudoku 9x9 (MRL)": ("sudoku-9x9_MRL", 81, "sudoku9x9/val"),
        "Sudoku 9x9 (attention)": ("sudoku-9x9_noMRL_attn", 81, "sudoku9x9/val"),
        "Sudoku 9x9 (attention, MRL)": ("sudoku-9x9_MRL_attn", 81, "sudoku9x9/val"),
    }
    _mn, _gs, _dp = _TASK_MAP[task_dropdown_s1.value]
    _lp = hf_hub_download(repo_id="davp-sapienza/trm-datasets", repo_type="dataset",
                          filename=f"{_dp}/labels.npy")
    _max_samples = np.load(_lp).shape[0]
    sample_idx_slider = mo.ui.slider(
        start=0, stop=_max_samples-1, step=1, value=0,
        label="Sample Index",
        show_value=True,
    )
    sample_idx_slider
    return sample_idx_slider


@app.cell
def _(DEVICE, TransformerBlock, TinyRecursiveModel, hf_hub_download, json, mo, np, torch):
    @mo.cache
    def run_grid_inference(task_name, sample_idx, n_steps):
        _TASK_MAP = {
            "Maze 5x5": ("maze5x5_noMRL", 81, "maze/maze5x5/test"),
            "Maze 5x5 (MRL)": ("maze5x5_MRL", 81, "maze/maze5x5/test"),
            "Maze 10x10": ("maze10x10_noMRL", 361, "maze/maze10x10/test"),
            "Maze 10x10 (MRL)": ("maze10x10_MRL", 361, "maze/maze10x10/test"),
            "Sudoku 9x9": ("sudoku-9x9_noMRL", 81, "sudoku9x9/val"),
            "Sudoku 9x9 (MRL)": ("sudoku-9x9_MRL", 81, "sudoku9x9/val"),
            "Sudoku 9x9 (attention)": ("sudoku-9x9_noMRL_attn", 81, "sudoku9x9/val"),
            "Sudoku 9x9 (attention, MRL)": ("sudoku-9x9_MRL_attn", 81, "sudoku9x9/val"),
        }
        _mn, _gs, _dp = _TASK_MAP[task_name]

        _cp = hf_hub_download(repo_id="davp-sapienza/trm-checkpoints", filename=f"{_mn}.json")
        _kp = hf_hub_download(repo_id="davp-sapienza/trm-checkpoints", filename=f"{_mn}.pt")
        _ip = hf_hub_download(repo_id="davp-sapienza/trm-datasets", repo_type="dataset", filename=f"{_dp}/inputs.npy")
        _lp = hf_hub_download(repo_id="davp-sapienza/trm-datasets", repo_type="dataset", filename=f"{_dp}/labels.npy")

        with open(_cp) as _f:
            _cfg = json.load(_f)
        _ckpt = torch.load(_kp, map_location="cpu", weights_only=True)
        _sd = _ckpt["model_state_dict"]

        _hs = _cfg["hidden_size"]
        _bk = dict(num_heads=_cfg["num_heads"], hidden_size=_hs, expansion=_cfg["expansion"],
                   seq_len=_gs, num_registers=_cfg["registers"], use_mlp_t=_cfg["mlp_t"])
        _net = torch.nn.Sequential(TransformerBlock(**_bk), TransformerBlock(**_bk))
        _m = TinyRecursiveModel(hidden_size=_hs, vocab_size=_sd["prediction_head.weight"].shape[0],
                                network=_net, n=_cfg["n"], T=_cfg["T"], num_registers=_cfg["registers"])
        _m.load_state_dict(_sd)
        _m.to(DEVICE)
        _m.eval()

        _all_inp = torch.from_numpy(np.load(_ip)).long()
        _all_lab = torch.from_numpy(np.load(_lp)).long()
        _idx = sample_idx % len(_all_inp)
        _inp = _all_inp[_idx:_idx+1].to(DEVICE)
        _lab = _all_lab[_idx:_idx+1].to(DEVICE)

        _m.disable_tracking()
        with torch.no_grad():
            _logits = _m.predict(_inp, N_supervision=n_steps)

        return (_logits.argmax(-1).cpu().numpy(),
                _lab.cpu().numpy(),
                _gs)

    return run_grid_inference


@app.cell
def _(go, mo, np, run_grid_inference, sample_idx_slider, step_slider_s1, task_dropdown_s1):
    _t = task_dropdown_s1.value
    _si = sample_idx_slider.value
    _s = step_slider_s1.value
    _preds, _labels, _gs = run_grid_inference(_t, _si, _s)
    _p = _preds[0]
    _l = _labels[0]
    _side = int(np.sqrt(_gs))
    _p2d = _p.reshape(_side, _side)
    _l2d = _l.reshape(_side, _side)

    if "Maze" in _t:
        _cs = [[0, "#f0f0f0"], [0.2, "#333"], [0.4, "#4CAF50"], [0.6, "#2196F3"], [0.8, "#F44336"], [1, "#FFC107"]]
        _ck = dict(colorscale=_cs, zmin=0, zmax=5, showscale=False, xgap=1, ygap=1)
        _fp = go.Figure(go.Heatmap(z=_p2d, **_ck))
        _fp.update_layout(title=f"Prediction — Step {_s}", width=320, height=320,
                          margin=dict(l=5, r=5, t=40, b=5))
        _fp.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
        _fl = go.Figure(go.Heatmap(z=_l2d, **_ck))
        _fl.update_layout(title="Ground Truth", width=320, height=320,
                          margin=dict(l=5, r=5, t=40, b=5))
        _fl.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    else:
        _a = [dict(x=j, y=i, text=str(_p2d[i, j]), showarrow=False,
                   font=dict(color="white" if _p2d[i, j] > 0 else "#ccc", size=12))
              for i in range(_side) for j in range(_side)]
        _al = [dict(x=j, y=i, text=str(_l2d[i, j]), showarrow=False,
                    font=dict(color="white" if _l2d[i, j] > 0 else "#ccc", size=12))
               for i in range(_side) for j in range(_side)]
        _fp = go.Figure(go.Heatmap(z=_p2d, colorscale="Blues", showscale=False, xgap=1, ygap=1))
        _fp.update_layout(title=f"Prediction — Step {_s}", width=380, height=380,
                          margin=dict(l=5, r=5, t=40, b=5), annotations=_a)
        _fp.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
        _fl = go.Figure(go.Heatmap(z=_l2d, colorscale="Blues", showscale=False, xgap=1, ygap=1))
        _fl.update_layout(title="Ground Truth", width=380, height=380,
                          margin=dict(l=5, r=5, t=40, b=5), annotations=_al)
        _fl.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")

    _token_acc = (_p == _l).mean() * 100
    _correct = "Yes" if (_p == _l).all() else "No"
    mo.vstack([
        mo.hstack([mo.ui.plotly(_fp), mo.ui.plotly(_fl)], gap=1),
        mo.md(f"**Token Accuracy:** `{_token_acc:.1f}%` | **Correct:** `{_correct}`"),
    ])
    return


# ─── §3.1  Effect of MRL ───────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-mrl"></a>
        ### §3.1  Effect of MRL

        As shown in the accuracy tables above, MRL yields virtually identical performance to the full latent representation, except for the Sudoku 9x9 MLP-Mixing configuration.
        MRL can therefore serve as a diagnostic probe for estimating the intrinsic dimensionality of the reasoning manifold without interfering with the deep supervision mechanism.
        Even when considering only the first few dimensions of $y$, the solution can be approximated effectively. For Maze $5\times5$, *using only the first dimension*, Token Accuracy remains around 83% (see the results table below).
        Empirical trials with acceleration-based early stopping revealed identical stopping patterns for MRL-enhanced and standard TRMs.
        Given this equivalence, the early stopping experiments shown above focus exclusively on non-MRL models.
        """
    )
    return


@app.cell
def _(mo):
    mrl_model_map = {
        "Maze 5×5 (MRL)": ("maze5x5_MRL", 81, "Maze 5x5"),
        "Maze 10×10 (MRL)": ("maze10x10_MRL", 361, "Maze 10x10"),
        "Sudoku 9×9 (MRL)": ("sudoku-9x9_MRL", 81, "Sudoku 9x9"),
        "Sudoku 9×9 (MRL, attn)": ("sudoku-9x9_MRL_attn", 81, "Sudoku 9x9 (attention)"),
    }
    mrl_dataset_dropdown = mo.ui.dropdown(
        options=list(mrl_model_map.keys()), value="Maze 5×5 (MRL)", label="MRL Model"
    )
    mrl_dataset_dropdown # type: ignore
    return mrl_dataset_dropdown, mrl_model_map


@app.cell
def _(DEVICE, TransformerBlock, TinyRecursiveModel, hf_hub_download, json, mo, np, torch):
    @mo.cache
    def compute_mrl_accuracy(mrl_label, model_map):
        _mn, _gs, _tr = model_map[mrl_label]
        _cp = hf_hub_download(repo_id="davp-sapienza/trm-checkpoints", filename=f"{_mn}.json")
        _kp = hf_hub_download(repo_id="davp-sapienza/trm-checkpoints", filename=f"{_mn}.pt")
        _DSP = {"Maze 5x5": "maze/maze5x5/test", "Maze 10x10": "maze/maze10x10/test",
                "Sudoku 9x9": "sudoku9x9/val", "Sudoku 9x9 (attention)": "sudoku9x9/val"}
        _ip = hf_hub_download(repo_id="davp-sapienza/trm-datasets", repo_type="dataset",
                              filename=f"{_DSP[_tr]}/inputs.npy")
        _lp = hf_hub_download(repo_id="davp-sapienza/trm-datasets", repo_type="dataset",
                              filename=f"{_DSP[_tr]}/labels.npy")
        with open(_cp) as _f:
            _cfg = json.load(_f)
        _ck = torch.load(_kp, map_location="cpu", weights_only=True)
        _sd = _ck["model_state_dict"]
        _hs = _cfg["hidden_size"]
        _bk = dict(num_heads=_cfg["num_heads"], hidden_size=_hs, expansion=_cfg["expansion"],
                   seq_len=_gs, num_registers=_cfg["registers"], use_mlp_t=_cfg["mlp_t"])
        _net = torch.nn.Sequential(TransformerBlock(**_bk), TransformerBlock(**_bk))
        _m = TinyRecursiveModel(hidden_size=_hs, vocab_size=_sd["prediction_head.weight"].shape[0],
                                network=_net, n=_cfg["n"], T=_cfg["T"], num_registers=_cfg["registers"])
        _m.load_state_dict(_sd)
        _m.to(DEVICE)
        _m.eval()
        _inp = torch.from_numpy(np.load(_ip)).long()[:200].to(DEVICE)
        _lab = torch.from_numpy(np.load(_lp)).long()[:200].to(DEVICE)
        _dims = [d for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if d <= _hs]
        _accs = []
        for _d in _dims:
            with torch.no_grad():
                _lg = _m.predict(_inp, N_supervision=16, W_slice=_d)
                _accs.append((_lg.argmax(-1) == _lab).float().mean().item() * 100)
        return np.array(_dims), np.array(_accs)

    return compute_mrl_accuracy


@app.cell
def _(compute_mrl_accuracy, go, mo, mrl_dataset_dropdown, mrl_model_map):
    _lb = mrl_dataset_dropdown.value
    _dims, _accs = compute_mrl_accuracy(_lb, mrl_model_map)

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_dims, y=_accs,
        marker=dict(color=_accs, colorscale="viridis", showscale=True,
                    colorbar=dict(title="Acc %"), line=dict(width=1, color="white")),
        hovertemplate="Dim: %{x}<br>Acc: %{y:.1f}%<extra></extra>"))
    _fig.update_layout(
        title=f"MRL Token Accuracy — {_lb}",
        xaxis=dict(title="Latent Dimension", type="log",
                   tickvals=_dims.tolist(), ticktext=[str(int(d)) for d in _dims]),
        yaxis=dict(title="Token Accuracy (%)", range=[0, 105]),
        width=700, height=380, margin=dict(l=10, r=10, t=50, b=10), bargap=0.15)

    _best = next(((int(d), float(a)) for d, a in zip(_dims, _accs) if a >= float(_accs[-1]) * 0.95), (0, 0))
    _fd = int(_dims[-1])
    _pct = _best[0] / _fd * 100 if _best[0] else 0

    mo.hstack([
        mo.ui.plotly(_fig),
        mo.callout(
            mo.md(f"""
            **Key Insight — {_lb}**

            Using just **{_best[0]}** of **{_fd}** dimensions ({_pct:.1f}%) achieves **{_best[1]:.1f}%** accuracy — over 95% of full performance.

            The reasoning manifold has **low intrinsic dimensionality**, validating MRL as an effective compression strategy.
            """),
            kind="success",
        ),
    ], gap=2, widths=[1.3, 0.7])
    return


# @app.cell
# def _(mo):
#     mo.md(
#         r"""
#         #### Results without early stopping

#         Below are reported the results obtained when using different supervision steps ($N_{\text{sup}}$). Maze tasks appear to be solvable with just one supervision step while Sudoku requires at least $N_{\text{sup}} = 2$.

#         <a id="tab-token"></a>
#         #### Token Accuracy by MRL Dimension

#         | Task | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | d=64 | d=128 |
#         |------|-----|-----|-----|-----|------|------|------|-------|
#         | Maze 5×5 | 82.8 | 98.8 | 99.9 | 99.9 | 99.9 | — | — | — |
#         | Maze 10×10 | — | 99.7 | 100 | 100 | 100 | 100 | — | — |
#         | Sudoku 9×9 | — | — | 96.2 | 96.3 | 96.3 | — | — | — |
#         | Sudoku 9×9 (attn) | — | — | — | — | 98.7 | 98.7 | 98.7 | 98.7 |

#         <a id="tab-eval"></a>
#         #### Exact Accuracy by Supervision Steps

#         | $N_{\text{sup}}$ | Maze 5×5 MRL(4,8,16) | Maze 5×5 noMRL(16) | Maze 10×10 MRL(4,8,16,32) | Maze 10×10 noMRL(32) | Sudoku 9×9 MRL(4,8,16) | Sudoku 9×9 noMRL(16) | Sudoku 9×9 attn MRL(16,64,128) | Sudoku 9×9 attn noMRL(128) |
#         |---|---|---|---|---|---|---|---|---|
#         | 1 | 98% | 87% | 100% | 98% | 61% | 61% | 69% | 69% |
#         | 2 | 98% | 96% | 100% | 97% | 77% | 77% | 93% | 93% |
#         | 4 | 98% | 96% | 100% | 97% | 82% | 82% | 94% | 94% |
#         | 8 | 98% | 95% | 100% | 97% | 82% | 82% | 94% | 94% |
#         | 16 | 99% | 95% | 100% | 97% | 82% | 82% | 94% | 94% |
#         """
#     )
#     return



# ─── §3.2  Geometric Metrics ───────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-metrics"></a>
        ### §3.2  Difference Between Reasoning and Solutions

        Within each deep supervision step, the features of $z$ exhibit high-frequency oscillations across the internal $n$ computational cycles before eventually aligning with the solution state $y$.
        This behavior makes $y$ more suitable than $z$ for defining a stable stopping criterion.

        *The following metrics are computed on both $z$ and $y$ (averaged across tokens, subsampled to $N_{\text{sup}}=16$ steps):
        average magnitude of $\Delta^{(k)}$, average step orthogonality $\cos\angle(\Delta^{(k)}, \Delta^{(k-1)})$,
        and average normalized acceleration ([Equation 3](#eq-norm-acceleration)). The comparison clearly shows how $z$ oscillates more than $y$.*
        """
    )
    return


@app.cell
def _():
    _BASE = "https://huggingface.co/datasets/davp-sapienza/trm-compressed-trajectories/resolve/main/plots"
    GEO_PLOT_FILES = {
        "Maze 5×5 (no MRL)": f"{_BASE}/maze5x5_noMRL.png",
        "Maze 5×5 (MRL)": f"{_BASE}/maze5x5_withMRL.png",
        "Maze 10×10 (no MRL)": f"{_BASE}/maze10x10_noMRL.png",
        "Maze 10×10 (MRL)": f"{_BASE}/maze10x10_withMRL.png",
        "Sudoku 9×9 (no MRL)": f"{_BASE}/sudoku9x9_noMRL.png",
        "Sudoku 9×9 (attn)": f"{_BASE}/sudoku9x9_noMRL_attention.png",
        "Sudoku 9×9 (MRL)": f"{_BASE}/sudoku9x9_withMRL.png",
        "Sudoku 9×9 (MRL, attn)": f"{_BASE}/sudoku9x9_withMRL_attention.png",
    }
    return GEO_PLOT_FILES



@app.cell
def _(GEO_PLOT_FILES, mo):
    geo_dataset_dropdown = mo.ui.dropdown(
        options=list(GEO_PLOT_FILES.keys()), value="Maze 10×10 (no MRL)", label="Dataset")
    geo_dataset_dropdown # type: ignore
    return geo_dataset_dropdown


@app.cell
def _(GEO_PLOT_FILES, geo_dataset_dropdown, mo):
    _dn = geo_dataset_dropdown.value
    mo.vstack([
        mo.md(f"**{ _dn }**"),
        mo.image(src=GEO_PLOT_FILES[_dn]),
    ])
    return


# ─── §3.3  Two-Regime Updates (PCA 3D) ─────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-two-regime"></a>
        ### §3.3  Two-Regime Updates

        Examination of individual trajectories reveals two distinct computational phases.
        Once a macroscopic **coarse solution** emerges during the initial recursive steps, the network transitions into a localized **fine refinement** regime.
        This topological behavior provides empirical support for the two-scale latent dynamics framework proposed by [Pappone et al. (2025)](https://arxiv.org/abs/2509.23314), suggesting that TRMs refine solutions incrementally rather than repeatedly altering their semantic interpretation at each step.

        *This is visually demonstrated in the [Trajectory Clustering](#s3-clustering) plot below (§3.4) and in the acceleration curves of [§3.5](#s3-early-stopping).*
        """
    )
    return


@app.cell
def _(DATASET_FILES, PCA, hf_hub_download, mo, np):
    @mo.cache
    def compute_pca_3d(dataset_name, vis_mode, sample_idx):
        _fn = DATASET_FILES[dataset_name]
        _p = hf_hub_download(repo_id="davp-sapienza/trm-compressed-trajectories",
                             filename=_fn, repo_type="dataset")
        _y = np.load(_p)["y_space"]
        _ns, _hs = _y.shape[1], _y.shape[2]
        _ss = max(1, _ns // 16)

        if vis_mode == "global":
            _rng = np.random.RandomState(42)
            _idx = _rng.choice(len(_y), size=min(250, len(_y)), replace=False)
            _sub = _y[_idx][:, :min(16, _ns), :]
            _flat = _sub.reshape(-1, _hs)
            _pca = PCA(n_components=3, random_state=42)
            _pts = _pca.fit_transform(_flat).reshape(len(_idx), -1, 3)
        else:
            _single = _y[sample_idx % len(_y)][::_ss]
            _pca = PCA(n_components=3, random_state=42)
            _pts = _pca.fit_transform(_single)[None]
        return _pts

    return compute_pca_3d


# ─── §3.4  Trajectory Clustering (ARC-AGI) ─────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-clustering"></a>
        ### §3.4  Trajectory Clustering

        Simultaneous analysis of multiple trajectories from initialization to convergence reveals a clear topological organization.
        Latent representations do not diffuse chaotically; instead, they progressively **accumulate and converge toward well-defined attractor regions** within the manifold, indicating that the reasoning process evolves toward stable fixed points.
        This observation may be related to ideas underlying Deep Equilibrium Models (DEQ) ([Bai et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf)) and the Implicit Function Theorem (IFT) discussed by [Wang et al. (2025)](https://arxiv.org/abs/2506.21734).

        *The trajectories used for the following plots are collected during evaluation on the validation/test splits, with shape `Samples × Steps × Seq len × Hidden size`. At each supervision step, the representation $y$ is averaged across tokens to obtain a single global representation of the puzzle.
        A three-dimensional PCA projection is then fitted jointly on the representations from all trajectories and supervision steps. This yields a shared coordinate system in which trajectories can be visualized and directly compared within the same compressed space.
        Colors denote the supervision step, thereby highlighting the temporal progression of the representations.*
        """
    )
    return


@app.cell
def _(DATASET_FILES, mo):
    cluster_dataset_dropdown = mo.ui.dropdown(
        options=list(DATASET_FILES.keys()), value="ARC-AGI v1 (test)", label="Dataset",
    )
    cluster_dataset_dropdown # type: ignore
    return cluster_dataset_dropdown



@app.cell
def _(compute_pca_3d, go, mo, np, px, cluster_dataset_dropdown):
    _ds = cluster_dataset_dropdown.value
    _pts = compute_pca_3d(_ds, "global", 0)
    _nt, _st, _ = _pts.shape
    _cl = px.colors.sample_colorscale("plasma", np.linspace(0, 1, _st))

    _fig = go.Figure()
    # Linee colorate per step (identico a pca-3d.py)
    for _t in range(_st - 1):
        _xs, _ys, _zs = [], [], []
        for _i in range(_nt):
            _xs.extend([_pts[_i, _t, 0], _pts[_i, _t+1, 0], np.nan])
            _ys.extend([_pts[_i, _t, 1], _pts[_i, _t+1, 1], np.nan])
            _zs.extend([_pts[_i, _t, 2], _pts[_i, _t+1, 2], np.nan])
        _fig.add_trace(go.Scatter3d(
            x=_xs, y=_ys, z=_zs, mode="lines",
            line=dict(color=_cl[_t], width=1), opacity=0.06,
            showlegend=False, hoverinfo="skip"))

    # Scatter first step (tutti i punti iniziali)
    _fig.add_trace(go.Scatter3d(
        x=_pts[:, 0, 0], y=_pts[:, 0, 1], z=_pts[:, 0, 2],
        mode="markers", marker=dict(color=_cl[0], size=3, opacity=0.5),
        name="First step", showlegend=True))

    # Scatter last step (tutti i punti finali — convergono verso attractor regions)
    _fig.add_trace(go.Scatter3d(
        x=_pts[:, -1, 0], y=_pts[:, -1, 1], z=_pts[:, -1, 2],
        mode="markers", marker=dict(color=_cl[-1], size=3, opacity=0.7),
        name="Last step", showlegend=True))

    _fig.update_layout(
        title=f"ARC-AGI Clusters — {_ds}",
        scene=dict(xaxis_title="<b>PC1</b>", yaxis_title="<b>PC2</b>", zaxis_title="<b>PC3</b>",
                   aspectmode="cube",
                   camera=dict(eye=dict(x=1.0, y=1.0, z=0.5))),
        width=900, height=700, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
    )
    mo.ui.plotly(_fig)
    return


# ─── §3.5  Early Stopping ──────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-early-stopping"></a>
        ### §3.5  Early Stopping

        Given this geometry, a halting mechanism becomes theoretically justified.
        As shown in the interactive controls below, acceleration-based early stopping substantially reduces the average number of computational steps across all tasks without sacrificing accuracy (see the [accuracy tables in §3.1](#s3-mrl) for comparison).
        Crucially, the unnormalized acceleration provides a **more robust stopping signal** than the normalized variant ([Equations 2](#eq-acceleration) and [3](#eq-norm-acceleration)).
        During the initial phase, updates $\Delta^{(k)}$ are large, whereas near convergence $\Vert \Delta^{(k)} \Vert_2 \rightarrow 0$ (localized refinements).
        In this transition between the two latent regimes, the unnormalized acceleration cleanly drops to zero because successive updates become identical.
        Conversely, for the normalized metric, both the numerator and the denominator vanish simultaneously. As a result, the normalized acceleration curve plateaus with severe oscillations and fails to reliably signal convergence.

        *Two-hit check: halt when $a^{(k)} < \tau$ for two consecutive steps.*
        """
    )
    return


@app.cell
def _(NON_AGI_FILES, mo):
    es_dataset_dropdown = mo.ui.dropdown(
        options=list(NON_AGI_FILES.keys()), value="Maze 10×10 (no MRL)", label="Dataset")
    es_sample_slider = mo.ui.slider(start=0, stop=559, step=1, value=42, label="Sample", show_value=True)
    tau_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.02, label="Threshold τ", show_value=True)
    mo.hstack([es_dataset_dropdown, es_sample_slider, tau_slider], gap=2)
    return es_dataset_dropdown, es_sample_slider, tau_slider


@app.cell
def _(NON_AGI_FILES, hf_hub_download, mo, np):
    @mo.cache
    def compute_acceleration(dataset_name, sample_idx):
        _p = hf_hub_download(repo_id="davp-sapienza/trm-compressed-trajectories",
                             filename=NON_AGI_FILES[dataset_name], repo_type="dataset")
        _y = np.load(_p)["y_space"]
        _ns = _y.shape[1]
        _ss = max(1, _ns // 16)
        _sg = _y[sample_idx % len(_y)][::_ss]
        _d = np.diff(_sg, axis=0)
        _a, _na = [], []
        for _k in range(1, len(_d)):
            _diff = _d[_k] - _d[_k - 1]
            _v = np.linalg.norm(_diff)
            _a.append(_v)
            _na.append(_v / (np.linalg.norm(_d[_k]) + np.linalg.norm(_d[_k - 1]) + 1e-8))
        return np.arange(len(_a)) + 2, np.array(_a), np.array(_na)

    @mo.cache
    def compute_sample_halt(dataset_name, sample_idx, tau):
        _p = hf_hub_download(repo_id="davp-sapienza/trm-compressed-trajectories",
                             filename=NON_AGI_FILES[dataset_name], repo_type="dataset")
        _y = np.load(_p)["y_space"]
        _ns = _y.shape[1]
        _ss = max(1, _ns // 16)
        _sg = _y[sample_idx % len(_y)][::_ss]
        _d = np.diff(_sg, axis=0)
        _mx = len(_d) + 1
        _h = _mx
        _ps = False
        for _k in range(1, len(_d)):
            _v = np.linalg.norm(_d[_k] - _d[_k - 1])
            _sm = _v < tau
            if _sm and _ps:
                _h = _k + 1
                break
            _ps = _sm
        return _h, _mx

    return compute_acceleration, compute_sample_halt


@app.cell
def _(
    compute_acceleration, compute_sample_halt,
    es_dataset_dropdown, es_sample_slider, go, mo, np, px, tau_slider,
):
    _ds = es_dataset_dropdown.value
    _si = es_sample_slider.value
    _tau = tau_slider.value
    _st, _ac, _an = compute_acceleration(_ds, _si)
    _halt, _mx = compute_sample_halt(_ds, _si, _tau)
    _compute_saved = (_mx - _halt) / _mx * 100
    _cl = px.colors.sample_colorscale("turbo", np.linspace(0, 1, len(_st)))

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_st, y=_ac, mode="lines+markers", name="a(k) — Standard",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=6, color=_cl, line=dict(width=1, color="white")),
        hovertemplate="Step %{x}<br>a(k)=%{y:.4f}<extra></extra>"))
    _fig.add_trace(go.Scatter(
        x=_st, y=_an, mode="lines+markers", name="â(k) — Normalized",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=5, symbol="x", color=_cl),
        hovertemplate="Step %{x}<br>â(k)=%{y:.4f}<extra></extra>", yaxis="y2"))
    _fig.add_hline(y=_tau, line_dash="dot", line_color="#d62728", line_width=2,
                   annotation_text=f"τ = {_tau:.3f}", annotation_position="top right")
    _fig.update_layout(
        title=f"Latent Acceleration — {_ds} (sample #{_si})",
        xaxis=dict(title="Supervision Step", dtick=1),
        yaxis=dict(title=dict(text="a(k)", font=dict(color="#1f77b4"))),
        yaxis2=dict(title=dict(text="â(k)", font=dict(color="#ff7f0e")),
                    overlaying="y", side="right", range=[0, 2.2]),
        width=720, height=400, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)), hovermode="x unified",
    )

    mo.vstack([
        mo.ui.plotly(_fig),
        mo.hstack([
            mo.stat(f"{_halt}", label=f"Halt Step (max {_mx})"),
            mo.stat(f"{_compute_saved:.1f}%", label="Compute Saved"),
        ], gap=1),
        mo.md(rf"> Two consecutive steps with $a^{{(k)}} < \tau = {_tau:.3f}$ trigger early stopping."),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        <a id="tab-ea"></a>
        #### Early Stopping Results

        | Dataset | τ=0.5 Acc% | Halted% | Avg Steps | τ=0.8 Acc% | Halted% | Avg Steps | τ=1.0 Acc% | Halted% | Avg Steps |
        |---------|-----------|---------|-----------|-----------|---------|-----------|-----------|---------|-----------|
        | Maze 5×5 | 96 | 44 | 12.0 | 95 | 84 | 6.4 | 96 | 100 | 2.0 |
        | Maze 10×10 | 96 | 38 | 12.0 | 97 | 71 | 7.8 | 97 | 100 | 2.0 |
        | Sudoku 9×9 | 82 | 71 | 9.0 | 80 | 90 | 5.3 | 77 | 100 | 2.0 |
        | Sudoku 9×9 (attn) | 94 | 49 | 11.0 | 94 | 84.2 | 7.0 | 93 | 100 | 2.0 |

        *Accuracy remains stable while average inference steps drop dramatically — e.g. from 16 to ~6.4
        for Maze 5×5 at τ=0.8, a **60% reduction** with less than 1% accuracy loss.*
        """
    )
    return



# ─── §3.6  Sample Trajectories ─────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        <a id="s3-sample-traj"></a>
        ### §3.6  Sample Trajectories

        The following plots show the trajectory of a randomly selected sample, projected into a compressed 3D space, with the goal of visualizing the path followed by the latent solution across supervision steps.

        The trajectory of $y$ is averaged across tokens to obtain a global representation at each step, then subsampled every $4$ steps to retrieve the $y$ values corresponding to the $N_{\text{sup}}=16$ supervision steps.

        Unlike the clustering analysis, here the PCA is fitted only on the points of this individual trajectory, projecting it independently into a 3D space to visualize its specific path. Alongside the trajectory, we report the standard and normalized acceleration ([Equations 2](#eq-acceleration) and [3](#eq-norm-acceleration)), computed from consecutive differences $\Delta^{(k)}$, with the supervision step encoded by color in both plots.
        """
    )
    return


@app.cell
def _(DATASET_FILES, mo):
    sample_traj_dropdown = mo.ui.dropdown(
        options=list(DATASET_FILES.keys()), value="Maze 5×5 (no MRL)", label="Dataset")
    sample_traj_slider = mo.ui.slider(start=0, stop=559, step=1, value=42, label="Sample Index", show_value=True)
    mo.hstack([sample_traj_dropdown, sample_traj_slider], gap=2)
    return sample_traj_dropdown, sample_traj_slider


@app.cell
def _(DATASET_FILES, PCA, hf_hub_download, mo, np):
    @mo.cache
    def compute_sample_trajectory(dataset_name, sample_idx):
        _fn = DATASET_FILES[dataset_name]
        _p = hf_hub_download(repo_id="davp-sapienza/trm-compressed-trajectories",
                             filename=_fn, repo_type="dataset")
        _y = np.load(_p)["y_space"]

        _single_all = _y[sample_idx % len(_y)]
        _ns = _single_all.shape[0]
        _ss = max(1, _ns // 16)
        _single = _single_all[::_ss]
        _num_steps = _single.shape[0]

        _deltas = np.diff(_single, axis=0)
        _acc = []
        _norm_acc = []

        for _k in range(1, len(_deltas)):
            _delta_cur = _deltas[_k]
            _delta_prev = _deltas[_k - 1]
            _diff = _delta_cur - _delta_prev

            _acceleration = np.linalg.norm(_diff)
            _acc.append(_acceleration)

            _norm_cur = np.linalg.norm(_delta_cur)
            _norm_prev = np.linalg.norm(_delta_prev)
            _norm_acc.append(_acceleration / (_norm_cur + _norm_prev + 1e-8))

        _full_acc = [np.nan, np.nan] + _acc
        _full_norm_acc = [np.nan, np.nan] + _norm_acc

        _pca = PCA(n_components=3)
        _traj_3d = _pca.fit_transform(_single)

        return _traj_3d, np.arange(_num_steps), np.array(_full_acc), np.array(_full_norm_acc)

    return compute_sample_trajectory


@app.cell
def _(compute_sample_trajectory, go, mo, np, px, sample_traj_dropdown, sample_traj_slider):
    _ds = sample_traj_dropdown.value
    _si = sample_traj_slider.value
    _traj_3d, _steps, _acc, _norm_acc = compute_sample_trajectory(_ds, _si)
    _num_steps = len(_steps)

    _colors = px.colors.sample_colorscale("viridis", np.linspace(0, 1, _num_steps))

    _fig_3d = go.Figure()

    for _i in range(_num_steps - 1):
        _fig_3d.add_trace(go.Scatter3d(
            x=_traj_3d[_i:_i+2, 0],
            y=_traj_3d[_i:_i+2, 1],
            z=_traj_3d[_i:_i+2, 2],
            mode="lines",
            line=dict(color=_colors[_i], width=5),
            opacity=0.8,
            showlegend=False,
            hoverinfo="skip"
        ))

    _fig_3d.add_trace(go.Scatter3d(
        x=_traj_3d[:, 0],
        y=_traj_3d[:, 1],
        z=_traj_3d[:, 2],
        mode="markers",
        marker=dict(
            size=8,
            color=list(range(_num_steps)),
            colorscale="viridis",
            line=dict(color="black", width=1),
            showscale=True,
            colorbar=dict(title="Supervision Step", thickness=15, len=0.5)
        ),
        name="Steps",
        hovertemplate="Step %{customdata}<extra></extra>",
        customdata=list(range(_num_steps))
    ))

    _fig_3d.update_layout(
        title=f"3D Trajectory — {_ds} (sample #{_si})",
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
            aspectmode="cube"
        ),
        width=600,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    _fig_acc = go.Figure()

    for _i in range(2, _num_steps - 1):
        _fig_acc.add_trace(go.Scatter(
            x=_steps[_i:_i+2],
            y=_acc[_i:_i+2],
            mode="lines",
            line=dict(color=_colors[_i], width=4),
            opacity=0.8,
            showlegend=False,
            hoverinfo="skip"
        ))
        _fig_acc.add_trace(go.Scatter(
            x=_steps[_i:_i+2],
            y=_norm_acc[_i:_i+2],
            mode="lines",
            line=dict(color=_colors[_i], width=3, dash="dash"),
            opacity=0.7,
            showlegend=False,
            hoverinfo="skip",
            yaxis="y2"
        ))

    _fig_acc.add_trace(go.Scatter(
        x=_steps[2:],
        y=_acc[2:],
        mode="markers",
        marker=dict(
            size=10,
            color=list(range(2, _num_steps)),
            colorscale="viridis",
            line=dict(color="black", width=1),
            showscale=False
        ),
        name="Standard",
        hovertemplate="Step %{x}<br>a(k)=%{y:.4f}<extra></extra>"
    ))

    _fig_acc.add_trace(go.Scatter(
        x=_steps[2:],
        y=_norm_acc[2:],
        mode="markers",
        marker=dict(
            size=8,
            symbol="x",
            color=list(range(2, _num_steps)),
            colorscale="viridis",
            line=dict(color="black", width=1),
            showscale=False
        ),
        name="Normalized",
        yaxis="y2",
        hovertemplate="Step %{x}<br>â(k)=%{y:.4f}<extra></extra>"
    ))

    _fig_acc.update_layout(
        title=f"Acceleration — {_ds} (sample #{_si})",
        xaxis=dict(title="Supervision Step", dtick=1),
        yaxis=dict(title="a(k) — Standard (solid line)"),
        yaxis2=dict(
            title="â(k) — Normalized (dashed line)",
            overlaying="y",
            side="right",
            range=[-0.05, 2.05]
        ),
        width=600,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        hovermode="x unified"
    )

    mo.hstack([mo.ui.plotly(_fig_3d), mo.ui.plotly(_fig_acc)], gap=2)
    return


# ═══════════════════════════════════════════════════════════════
#  §4  CONCLUSION
# ═══════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(
        r"""
        ## §4  Conclusion and Future Work

        We investigated the latent dynamics of TRMs, identifying a two-scale geometric regime in which, once a coarse macroscopic solution is established, the trajectory enters a localized refinement phase.
        We further showed that the more stable unnormalized acceleration-based early stopping criterion reduces inference-time computation without sacrificing performance.
        Additionally, we demonstrated that MRL provides an effective tool for monitoring the intrinsic dimensionality of the reasoning space.
        Several directions remain open for future research:

        1. can the latent reasoning space learned on Sudoku be transferred to related tasks such as KenKen or Latin squares?
        2. can trajectories be accelerated through autoregressive generation or direct future-state prediction?
        3. in tasks admitting multiple solutions, can TRMs become truly probabilistic by returning multiple valid answers?
        4. can beam decoding over latent paths replace PTRM's brute-force parallel rollouts as a more structured and efficient probabilistic inference strategy?

        ---

        *Notebook by **Davide Perniconi** — alphaXiv × marino #2 Competition*
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion({
        "Appendix — Limitations": mo.md(
            r"""
            The primary limitation of this study introduces a stark computational trade-off between the inherent overhead of the TRM architecture and the hardware resources required to validate early stopping.
            Paradoxically, evaluating an early stopping criterion demands complex, high-dimensional reasoning problems that natively require a large number of supervision steps ($N_{\text{sup}}$) to justify the computational exit.
            However, the sequential nature of this recursive mechanism creates a severe computation bottleneck. Consequently, testing the maximum recurrence bounds proved computationally intractable within standard cloud environments like Google Colab or Kaggle.
            Furthermore, our geometric exploration was bounded by the public checkpoints released by the original authors. Because they exclusively provided pre-trained artifacts for the ARC-AGI benchmarks, other tasks like Sudoku 9x9 Extreme and Maze 30x30 Hard remained inaccessible for trajectory analysis.
            Finally, due to strict memory and storage constraints when logging massive sequences of intermediate hidden states across numerous steps, the collective study of ARC-AGI latent trajectories had to be restricted to a bounded subset of samples, scaled to the absolute maximum threshold permitted by the available hardware constraints, which nevertheless provided a sufficiently dense corpus to isolate and verify stable convergence behaviors.
            """
        ),
    })
