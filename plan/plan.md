# Learning Traversability Maps from Overhead Imagery for Drone Routing
# 從衛星影像學習可通行性地圖用於無人機路徑規劃：完整實作計畫

---

## Development Principles / 開發原則

### Fix the logic, never weaken the test / 修邏輯，不弱化測試

When a test fails because the system doesn't meet the expected behavior, **fix the underlying logic**, never weaken the assertion. Tests encode the specification — weakening them hides real deficiencies and erodes the system's guarantees over time.

**The rule:** If a clean synthetic input (hand-built 4×4 cost map, analytical Dubins path, known-translation correspondences) produces the wrong output, the system is wrong, not the test.

### Planning-facing evaluation is the thesis / 以規劃為導向的評估是全案主軸

**Segmentation metrics alone are not a success criterion.** Every perception change must be scored on downstream route-quality metrics (path success, excess cost, obstacle violation). If mIoU goes up by 2 points but the violation rate on `urban→rural` also goes up, that is a regression, not progress.

---

## 0. Challenge 總覽（中文）

### 這個 Project 在做什麼？

建立一條「**衛星影像 → 語意地圖 → 可通行性成本圖 → 路徑規劃**」的完整 geospatial autonomy pipeline，專門給無人機配送用。核心問題是：**光有好的 segmentation mIoU 不代表 planner 跑得出安全路徑**。這個計畫刻意把 perception 和 planner 綁在一起評估，讓所有的感知改動都用 routing 層的指標去驗證。

### Input / Output

- **Input**: LoveDA overhead RGB tile（urban 或 rural）、GT semantic mask、以及 MVP 中使用的 `(start_xy, goal_xy)` oracle map coordinates。
- **Output per tile**: 每一張圖產出
  - `semantic_mask`：8 類 LoveDA labels（`ignore, background, building, road, water, barren, forest, agriculture`）
  - `cost_map`：`H×W` 成本圖，lethal 類別（building / water）→ `inf`
  - `uncertainty_map`：每像素的 softmax entropy（MVP）；MC-dropout variance 是 post-MVP
  - `route`：grid A* 回傳的 waypoint list（MVP）
  - `route_metrics`：`path_success, excess_cost, violation_rate, lethal_fraction, runtime`

### MVP 與後續階段怎麼切？

| Stage | 目的 | 核心假設 | 主要產出 |
|-------|-----|---------|---------|
| **MVP / current roadmap** | Segmentation release + baseline routing | Oracle start/goal 已知，cost map 由 semantic rule 產生 | 57.3 mIoU gate、rule-based cost map、grid A*、route metrics table |
| **Post-MVP A** | Motion-planning depth | drone 有 curvature / dynamics 限制 | Hybrid A*、RRT* 家族、replanning |
| **Post-MVP B** | Cross-view localization | 移除 oracle start 假設 | retrieval、matching + PnP、pose-coupled planning |

### Baseline 是什麼？

**"SegFormer + constant-cost A*"** — 只跑最乾淨的 CE 訓練，把語意圖直接 threshold 成「可走 / 不可走」二值圖，再用 4-connected grid A* 規劃。沒有 domain-shift 評估、沒有 kinodynamic 約束、沒有 uncertainty。這個 baseline 在 `urban→urban` 上看起來不錯，但會在 `urban→rural` 和狹窄道路上崩掉 —— 這正是本計畫想展示的 gap。

### 我們提出的架構是什麼？

MVP 的核心設計：

1. **Domain-aware perception**：train + eval 都明確區分 urban / rural，confusion matrix 和 per-class IoU 都要分 scene 報。
2. **Rule-based traversability cost map**：LoveDA 沒有 GT cost map；predicted/oracle cost 都由 semantic mask 經同一套 deterministic policy 產生。
3. **Planning-facing evaluation**：planner 在 predicted cost 上規劃，但 route 放回 GT-semantics-derived oracle cost 上評估。
4. **Semantic uncertainty as a cost variant**：先用 softmax entropy / pessimistic top-2 inflation 測試 confidence-aware routing 是否降低 lethal violations。

### 怎麼 Evaluate？

| 評估面向 | 主要 metric | 為什麼重要 |
|---------|-----------|----------|
| **Segmentation** | mIoU / per-class IoU（特別 `road`）/ confusion matrix | 感知品質的基本面，但**單獨不夠** |
| **Route quality（核心）** | Path success、excess cost、obstacle violation rate、lethal-cell length fraction | 真正反映 autonomy 能不能用 |
| **Domain shift** | 上述所有指標 × `{urban→urban, rural→rural, urban→rural, rural→urban}` | 這是 operationally 最重要的 gap |
| **Uncertainty cost variant** | Violation-rate reduction from default → entropy/pessimistic cost | 證明 uncertainty 是有用的，不只是花俏 |
| **Post-MVP planning** | curvature feasibility、sampling-based planner comparison | 在 MVP 完成後再回答 motion-planning 深度問題 |
| **Post-MVP localization** | Recall@K、meter-level accuracy、route degradation vs. pose error | MVP 後再移除 oracle start 假設 |

---

## 1. Strategic Context / 策略背景

The project brief ([doc/drone_routing_project_brief.pdf](../doc/drone_routing_project_brief.pdf)) framed this as a 12-day sprint on LoveDA segmentation + grid A*. The revised roadmap keeps that as the MVP and adds a stricter segmentation release path plus planning-facing evaluation. Advanced planners and localization remain valuable post-MVP research extensions, but they should not block the first credible perception-to-routing result.

| Requirement | What it means | Architecture response |
|-------------|--------------|----------------------|
| **Perception must be planning-aware** | mIoU alone can rise while `road` IoU and route violations worsen | Every eval pass computes route-quality metrics alongside segmentation |
| **Perception uncertainty is a planning input, not an afterthought** | Planner should avoid cells where the model itself is unsure | MVP cost map accepts `semantic_cost + λ·H(p)` and pessimistic top-2 lethal inflation |
| **Motion constraints matter later** | A drone cannot teleport between grid cells | Hybrid A* / RRT* are post-MVP extensions after the grid-A* benchmark exists |
| **Domain shift is the real deployment condition** | Urban↔rural class priors and scale differ significantly | All metrics reported in 4 settings: `u→u`, `r→r`, `u→r`, `r→u` |

---

## 2. System Architecture / 系統架構

This diagram shows the long-term architecture. The revised MVP roadmap implements the SegFormer -> rule-based cost map -> grid A* -> route-metrics subset first; Hybrid A*, RRT*, MC-dropout, replanning, and localization are post-MVP extensions.

```
Overhead RGB tile  (LoveDA urban / rural)
  ↓
[Preprocessor]  ─── patch, normalize, scene metadata attached via ComposeDict + AddMetadata
  ↓
[SegFormer]  ─── HuggingFace SegformerForSemanticSegmentation (b0/b1), logits @ H/4×W/4 → bilinear upsample
  ↓
[Per-pixel softmax]  ─── probs, argmax mask, entropy map, (optional) MC-dropout variance
  ↓
[Cost Map Builder]  ◄── RULE-BASED + UNCERTAINTY
  │  semantic_to_cost(mask, yaml) + λ·H(p) + pessimistic top-2 lethal inflation
  │  lethal classes → inf (building, water)
  ↓
[Start/Goal Sampler]  ─── valid pairs on low-cost cells, min-distance, oracle-A*-must-succeed
  ↓
[Planner]  ◄── PROGRESSIVE STACK
  │  ┌─ Grid A* (4/8-conn)               — Phase 1 baseline
  │  ├─ Hybrid A* (x,y,θ) + Dubins/RS    — curvature-bounded
  │  ├─ RRT* / Informed RRT*             — sampling-based comparison
  │  └─ Kinodynamic RRT*                  — primitives in (x,y,θ,v)
  ↓
[Observation-Triggered Replanner]  ─── simulate traversal, replan at high-uncertainty cells
  ↓
[Route-Quality Evaluator]  ─── success, excess cost, violation rate, lethal fraction, curvature, runtime
  ↓
[Phase 3: Cross-View Localizer]  ─── replaces oracle start_xy
    University-1652 / SUES-200 retrieval → SuperGlue/LoFTR matching → PnP → 2D pose
```

### Why a progressive planner stack, not a single planner

Each planner is a **controlled ablation** over the previous one:

| Planner | What it adds over prev | What question it answers |
|---------|-----------------------|-------------------------|
| Grid A* | — | Is the cost map usable at all? |
| Hybrid A* | `θ` state + curvature bound | Does kinodynamic feasibility change which routes win? |
| RRT* | Sampling, asymptotic optimality | Is grid resolution masking planning errors? |
| Informed RRT* | Ellipsoidal sample shaping | How much of RRT*'s cost comes from wasted samples? |
| Kinodynamic RRT* | Primitive-based edges | Does sampling + curvature compose cleanly? |

The **benchmark matrix itself is a deliverable** — nobody gets to publish a uniform "my planner is best" headline; the table is the result.

---

## 3. Repository Structure / 資料夾結構

```
Learning-Traversability-Maps-from-Overhead-Imagery-for-Drone-Routing/
├── src/loveda_project/
│   ├── data.py                    # ✅ DONE — LoveDA loaders, ConcatDataset, metadata collate
│   ├── transforms.py              # ✅ DONE — ComposeDict paired image/mask augs
│   ├── modeling.py                # ✅ DONE — SegFormer b0/b1 builder with HF pretrained
│   ├── losses.py                  # ✅ DONE — CE + multiclass Dice
│   ├── metrics.py                 # ✅ DONE — mIoU, per-class IoU, CM saver
│   ├── filename_utils.py          # ✅ DONE
│   ├── inference.py               # ✅ DONE — frozen checkpoint inference + entropy
│   ├── costmap.py                 # 🔲 PR 09/15 — semantic→cost + uncertainty variants
│   ├── uncertainty.py             # 🔲 PR 15 — entropy helper
│   ├── route_metrics.py           # 🔲 PR 12 — path success, excess cost, violation rate
│   ├── sampling.py                # 🔲 PR 11 — valid start/goal pairs
│   ├── benchmark.py               # 🔲 PR 16 — perception→routing benchmark schema
│   └── planners/
│       └── astar.py               # 🔲 PR 10 — grid A* MVP planner
├── scripts/
│   ├── day1_day2_setup.py         # ✅ DONE — data download + histogram + sample viz
│   ├── train_segformer.py         # ✅ DONE — training entry point, supports warmup+cosine, AMP, W&B
│   ├── eval_segformer.py          # ✅ PR 02 — frozen checkpoint eval; 🔲 PR 06 — sliding-window/TTA
│   ├── run_segmentation_sweep.py  # 🔲 PR 07 — release ablation sweep
│   ├── verify_segmentation_release.py # 🔲 PR 08 — 57.3 mIoU gate
│   ├── visualize_costmap.py       # 🔲 PR 09 — cost-map sanity figures
│   ├── run_phase1_routing.py      # 🔲 PR 13 — end-to-end routing script
│   ├── run_domain_routing_eval.py # 🔲 PR 14 — urban/rural routing matrix
│   ├── run_planner_benchmark.py   # 🔲 PR 16 — checkpoint × cost-map × domain sweep
│   ├── mine_failure_modes.py      # 🔲 PR 17 — failure taxonomy
│   ├── make_failure_panels.py     # 🔲 PR 17 — qualitative overlays
│   └── verify_final_artifacts.py  # 🔲 PR 18 — final claim consistency
├── configs/
│   ├── segmentation/              # 🔲 PR 07 — release sweep configs
│   ├── costmap/                   # 🔲 PR 09/15 — YAML cost policies
│   └── routing/                   # 🔲 PR 14/16 — domain and benchmark configs
├── tests/                         # ✅ PR 01 — pytest bootstrap + segmentation-core regressions
│   ├── test_smoke.py              # ✅ package import smoke test
│   ├── test_segmentation_core.py  # ✅ transforms/loss/metrics/modeling tests
│   ├── test_costmap.py            # 🔲 PR 09 — planned
│   ├── test_astar.py              # 🔲 PR 10 — planned
│   └── ...                        # one file per PR, mostly synthetic/hand-built inputs
├── data/                          # ✅ DONE — LoveDA tiles downloaded
├── outputs/
│   ├── day1_day2/                 # ✅ histograms, sample overlays
│   ├── day3_day5/                 # ✅ SegFormer constant-LR baseline
│   ├── day3_day5_warmcos/         # ✅ SegFormer warmup+cosine
│   └── phase1_routing/            # 🔲 PR 13+ outputs (route JSON + figures)
├── plan/
│   ├── plan.md                    # this file
│   └── CLAUDE_from_contextlens.md # format reference
├── RUN.md                         # ✅ canonical training commands
├── CLAUDE.md                      # ✅ per-chat context
└── requirements.txt               # ✅ DONE
```

---

## 4. Current Status & Performance / 現有進度與效能

### ✅ Done

- **Data pipeline** (`src/loveda_project/data.py`): `WrappedLoveDAScene` per `(split, scene)` → `ConcatDataset`, custom `_collate_samples` preserves `split_name` / `scene_name` lists, `LoveDAConfig` dataclass, class histogram + sample-grid visualizers.
- **Transforms** (`src/loveda_project/transforms.py`): paired `ComposeDict` with `EnsureTensorTypes`, `RandomCropPair`, `RandomHorizontalFlipPair`, `RandomVerticalFlipPair`, ImageNet normalization. Val pipeline uses `CenterCropPair`.
- **Model** (`src/loveda_project/modeling.py`): SegFormer b0/b1 from `nvidia/segformer-b{0,1}-finetuned-ade-512-512` with `ignore_mismatched_sizes=True` to remap the head to 8 classes.
- **Loss** (`src/loveda_project/losses.py`): `SegmentationCriterion` with `ce` or `ce_dice`; Dice zeroes the ignore channel at both pixel and class level.
- **Training loop** (`scripts/train_segformer.py`): AMP, 3 scheduler modes (`none`, `cosine-only`, `warmup+cosine`, stepped per-batch), W&B logging, best-mIoU checkpoint + normalized confusion-matrix PNG.
- **Metrics** (`src/loveda_project/metrics.py`): `SegmentationMeter` confusion-matrix-based mIoU / per-class IoU / pixel accuracy, plus confusion-matrix plot and metrics-JSON savers.
- **Frozen inference** (`src/loveda_project/inference.py`, `scripts/eval_segformer.py`): reusable no-grad SegFormer prediction API returning upsampled logits, probabilities, masks, and entropy, plus a checkpoint eval script that emits metrics JSON and confusion matrices.
- **Test bootstrap** (`tests/`): pytest import smoke test plus synthetic regression tests for `EnsureTensorTypes`, Dice ignore handling, `SegmentationMeter`, and SegFormer B0 model construction. Verified with `conda run -n loveda env PYTHONPATH=src python -m pytest tests`.

### 📊 Best recorded performance (urban + rural combined)

| Run | Loss | Dice w | Scheduler | LR | Epochs | **mIoU** | Pix Acc |
|-----|------|--------|-----------|-----|--------|----------|---------|
| `b0_pt_cedice_w0p25_patch512` | CE+Dice | 0.25 | constant | 6e-5 | 25 | **0.5110** | 0.6914 |
| `b0_pt_cedice_w0p25_warmcos4_lr8en5_patch512` | CE+Dice | 0.25 | warmup+cosine (warmup=5) | 8e-5 | 35 | **0.5119** | 0.6901 |

**Per-class IoU (best run, `day3_day5_warmcos`):**

| Class | IoU | Train pixel share | Notes |
|-------|-----|-------------------|-------|
| water | **0.683** | 6.6% | easiest, distinctive color |
| building | **0.621** | 11.1% | strong structural cue |
| road | **0.546** | **5.3%** | **routing-critical but only mediocre** |
| background | 0.531 | 35.5% | dominant class, absorbs confusion |
| agriculture | 0.509 | 20.3% | — |
| forest | 0.403 | 16.0% | confused with agriculture |
| **barren** | **0.291** | 5.3% | **worst — heavy confusion into background/agriculture** |

**Diagnosis from the confusion matrix:**
- The "column 1 = background" column is massive for every non-background class. **The model over-predicts background**, which steals pixels from minority classes (especially `road`, `barren`, `forest`).
- `road` IoU of 0.55 is the **single biggest blocker** for the whole thesis: routing metrics will be noisy until road is reliably segmented. Every predicted-road pixel that actually is background is a phantom corridor; every real road pixel that becomes background is a severed corridor.
- `barren` at 0.29 is essentially a failure class — confused with `background` and `agriculture`, both of which have similar earth tones.
- Two runs ≈ 0.51 mIoU with very different schedules suggests we've saturated what a vanilla **B0 + CE+Dice + RGB-only + basic augmentation** recipe can deliver.

### 🔲 Not yet implemented (covered by PRs below)

`semantic_to_cost`, grid A*, start/goal sampling, route-quality evaluation, uncertainty-aware cost variants, segmentation sweep tooling, release-gate tooling, benchmark scripts, and final artifact verification. Localization and advanced planners are post-MVP extensions, not blockers for the current release path.

---

## 5. Improvement Opportunities (from current performance) / 效能改善機會

Based on the confusion-matrix analysis above, the following improvements are **mandatory segmentation-release upgrades before claiming the final resume/report mIoU**. They are implemented early in the revised PR roadmap (PR 03–08), and the final headline number must be backed by a reproducible run artifact.

### 57.3 mIoU Release Gate / 57.3 mIoU 發布門檻

`57.3% mIoU` is a **hard claim gate**, not a guaranteed training outcome. The plan can guarantee that this number is not claimed unless a reproducible run reaches it.

- **Baseline**: current best `mean_iou = 0.5119` from the SegFormer-B0 warmup+cosine run.
- **Target**: final selected segmentation run must satisfy `mean_iou >= 0.5730`, i.e. at least `+6.1` points over the current baseline.
- **Official metric**: LoveDA validation set, urban+rural combined, ignore class excluded, computed with the same `SegmentationMeter` used by the existing `best_metrics.json`.
- **Required evidence**: `best_metrics.json`, best checkpoint, run config, confusion matrix, W&B/run log, and the exact reproduction command recorded in `RUN.md`.
- **Resume/report rule**: if the release gate fails, the resume must use the measured best mIoU or the wording `targeting 57.3%`, not `reaching 57.3%`.

| # | Improvement | Expected win | Risk | Rationale |
|---|-------------|-------------|------|-----------|
| **I1** | **Class-balanced / focal loss for minority classes** | +3–6 mIoU on `road`, `barren`, `forest`; slight drop on `background` | Over-weight can destabilize training | Directly addresses the "over-predict background" failure mode. Use inverse-frequency or effective-number-of-samples weighting; focal loss (γ=2) as alternative. |
| **I2** | **Stronger augmentation: ColorJitter + RandomScale + photometric ops** | +2–4 mIoU, better rural→urban generalization | Longer epoch time | LoveDA urban/rural differ mostly in color and scale; photometric augmentation is the cheapest domain-robustness fix. |
| **I3** | **Upgrade backbone to SegFormer-B1 / B2** | +3–5 mIoU (published B1 ≈ 52–55% mIoU class range on LoveDA) | 2–4× VRAM and training time | Current B0 is near its intrinsic ceiling at ~0.51. B1 is the standard step-up; B2 is a stretch. |
| **I4** | **Multi-scale / sliding-window inference (TTA)** | +1–3 mIoU, free at eval time | Slower eval | LoveDA tiles are 1024×1024 but we train on 512-crops; full-tile sliding window recovers context the patch classifier loses. |
| **I5** | **Boundary / lovász loss for thin structures** | +2–4 `road` IoU specifically | Harder to train; needs care | Roads are thin and topology-sensitive. Boundary-aware losses (e.g. Lovász-Softmax) explicitly reward thin-class boundary accuracy. |

**Strategy**: I1 and I2 land first as PR 03–04 because they directly address the background-overprediction failure mode. I3 lands as PR 05 through B2 + gradient accumulation. I4 lands as PR 06 and should be the official release-gate evaluation mode if it helps. I5 is tracked as a possible follow-up if `road` IoU remains the blocker after PR 03–06. PR 07–08 make the final mIoU claim auditable instead of relying on a hand-picked terminal log.

---

## 6. Module Implementation Details / 模組實作細節

### 6.1 Segmentation Release Path (`PR 02–08`)

The current trained baseline is `SegFormer-B0 + CE+Dice + warmup+cosine`, with `mean_iou = 0.5119`. The release path adds:

- frozen inference with upsampled logits, probabilities, masks, and entropy (`PR 02`);
- class-balanced / focal losses for minority classes (`PR 03`);
- stronger photometric and scale augmentation (`PR 04`);
- SegFormer-B2 plus gradient accumulation (`PR 05`);
- sliding-window / multi-scale evaluation (`PR 06`);
- config-driven sweep and result table (`PR 07`);
- `57.3% mIoU` release-gate verifier (`PR 08`).

The official segmentation claim is based on `best_metrics.json`, not a terminal log. If `mean_iou < 0.5730`, the project reports the measured best value or says `targeting 57.3%`.

### 6.2 Cost Map Builder (`src/loveda_project/costmap.py`, PR 09 + 15)

**Inputs**: `mask: LongTensor[H, W]`, optional `probs: FloatTensor[C, H, W]`, and a YAML cost policy.

**Important wording**: LoveDA does not provide real ground-truth cost maps. The oracle map is a **semantic proxy** produced from the GT semantic mask using the same deterministic policy used for predicted masks.

| Class | Default cost | Rationale |
|-------|--------------|-----------|
| `road` | 1.0 | preferred corridor |
| `background`, `barren`, `agriculture` | 3.0 | traversable but not preferred |
| `forest` | 5.0 | slower, higher risk |
| `building`, `water` | `inf` | lethal semantic regions |
| `ignore` | fill or mark by config | avoid silently rewarding unknown pixels |

**Entropy penalty (PR 15)**: `cost <- semantic_cost + lambda * H(p)` where `H(p) = -sum_c p_c * log(p_c + eps)` with `eps = 1e-12`. The `eps` shift prevents `0 * log(0) = NaN` for one-hot distributions; equivalent to using `torch.special.xlogy`.

**Pessimistic top-2 lethal (PR 15)**: if `argmax` is traversable but the runner-up is `building` or `water` with probability above `tau`, inflate cost by `alpha`.

### 6.3 Planner and Query Generation (`PR 10–11`)

**Grid A* (PR 10)** is the MVP planner: standard 2D A* over an `H x W` cost array, 4- or 8-connectivity, finite-cost cells only, and return payload `(path_coords, total_cost, num_expanded)`.

**Start/goal sampler (PR 11)** samples route queries on the oracle cost map. Endpoints must be finite-cost cells, separated by `min_dist`, and optionally reachable by oracle A*. This prevents bad endpoint sampling from being confused with perception failure.

Advanced planners (Hybrid A*, RRT*, kinodynamic RRT*) are post-MVP extensions after the perception-to-routing benchmark exists.

### 6.4 Route-Quality Metrics (`src/loveda_project/route_metrics.py`, PR 12)

Routes are planned on the predicted cost map, then scored on the oracle cost map derived from GT semantics.

| Metric | Formula / definition | What it catches |
|--------|----------------------|-----------------|
| **Path success** | `1[planner returns any path]` | disconnected predicted map, over-aggressive lethal costs |
| **Excess path cost** | `cost_oracle(pred_route) / cost_oracle(oracle_route) - 1`, computed only on **non-violating** routes (see below) | operational suboptimality under the oracle semantic policy |
| **Violation rate** | `1[any lethal oracle cell on pred_route]` | semantic safety failure |
| **Lethal fraction** | `lethal_cells_on_pred_route / route_length` | severity of lethal crossing |
| **Runtime** | wall-clock per query | planner practicality |

**Aggregation rule for `excess_cost` under violations.** Lethal cells have `oracle_cost = inf`, so any route that crosses one would yield `excess_cost = inf` and poison mean/median aggregation. The headline `excess_cost` is therefore reported **conditional on `violation_rate = 0`**, with `n_valid_routes` published alongside. Violating routes contribute to `violation_rate` and `lethal_fraction` but are excluded from `excess_cost` aggregation. The benchmark JSON schema (§6.5) must include `n_valid_routes` so a low `excess_cost` over a tiny valid subset is not mistaken for safe routing.

The project should say "semantic lethal-region violation", not "real-world collision", because LoveDA has no 3D geometry, altitude, or physical collision labels.

### 6.5 Benchmark and Reporting (`PR 13–18`)

- `PR 13`: one-checkpoint Phase-1 routing script.
- `PR 14`: four-domain routing matrix.
- `PR 16`: checkpoint x cost-policy x domain benchmark harness.
- `PR 17`: failure mining and qualitative overlays.
- `PR 18`: final artifact verifier so README/report claims match JSON.

**Checkpoint resolution convention.** PR 07's sweep emits `<sweep-dir>/winner.json` listing the best run per `train_scenes` axis (`combined`, `urban-only`, `rural-only`), and copies/symlinks each winner's `best_model.pth` to `<sweep-dir>/winners/{combined,urban-only,rural-only}/best_model.pth`. All downstream PRs (13, 14, 16, 18) reference these stable paths (or a `--sweep-dir` + `--winner` pair that resolves through `winner.json`) rather than guessing per-run subdirectory names. The `winner.json` also records the run config and `best_metrics.json` path for full traceability.

The final table must report at least:

```text
checkpoint, cost_policy, domain_setting,
mean_iou, road_iou,
route_success, excess_cost, n_valid_routes, lethal_fraction, violation_rate, runtime_ms
```

`excess_cost` is aggregated over `n_valid_routes` (non-violating successful routes only — see §6.4); the field is present so reviewers can spot a low excess-cost computed over a tiny valid subset.

---

## 7. Evaluation Plan / 評估計畫

### 7.1 Segmentation metrics (in-domain)

- **mIoU** over non-ignore classes; **per-class IoU** with `road`, `building`, `water` called out; **pixel accuracy**; **row-normalized confusion matrix** at the best checkpoint.
- **Release gate**: `mean_iou >= 0.5730` is required before the report/resume can claim `57.3% mIoU`.
- **Evidence**: `best_metrics.json`, checkpoint path, run config, confusion matrix, exact command, and sweep table row.

### 7.2 Route-quality metrics (THE HEADLINE)

For each route query:

- build `predicted_cost` from the model prediction;
- build `oracle_cost` from the GT semantic mask;
- plan with grid A* on `predicted_cost`;
- compute `oracle_route` with grid A* on `oracle_cost`;
- score the predicted route on `oracle_cost`.

Report:

- `path_success`
- `excess_cost`
- `violation_rate`
- `lethal_fraction`
- `runtime_ms`

### 7.3 Domain-shift matrix

All route-quality metrics are reported in **four settings**:

| Setting | Meaning |
|---------|---------|
| `urban->urban` | urban-only-trained checkpoint evaluated on urban validation tiles |
| `rural->rural` | rural-only-trained checkpoint evaluated on rural validation tiles |
| `urban->rural` | urban-only checkpoint evaluated on rural validation tiles |
| `rural->urban` | rural-only checkpoint evaluated on urban validation tiles |

The four cells require **three trained checkpoints**: `combined` (urban+rural, used as the headline release model), `urban-only`, and `rural-only`. PR 07's `release.yaml` must include all three `train_scenes` configurations. The urban-only and rural-only checkpoints exist solely to populate the domain-shift matrix and are not subject to the `57.3% mIoU` release gate (which applies to the combined checkpoint).

### 7.4 Benchmark table

One row per `(checkpoint, cost_policy, domain_setting)`:

```text
checkpoint_id          # one of: combined, urban-only, rural-only (resolved via winner.json)
cost_policy
domain_setting
mean_iou
road_iou
route_success
excess_cost            # aggregated over non-violating routes only (see §6.4)
n_valid_routes         # number of non-violating successful routes contributing to excess_cost
violation_rate
lethal_fraction
runtime_ms
num_tiles
num_route_pairs
```

The core comparison is:

| Claim | Supporting comparison |
|-------|----------------------|
| Better perception helps routing | baseline B0 vs best release checkpoint on the same cost policy |
| Road IoU matters | route metrics plotted against `road_iou` across checkpoints |
| Uncertainty helps safety | default cost vs entropy / pessimistic cost on `violation_rate` and `lethal_fraction` |

### 7.5 Qualitative checks

Failure panels should show the RGB image, GT mask, predicted mask, predicted cost, oracle cost, predicted route, oracle route, and highlighted lethal crossings. These figures are supporting evidence; JSON metrics are the source of truth.

---

## 8. Revised PR Roadmap / 重新制定 PR 路線圖

**Current codebase state**: the repository already has a working LoveDA data pipeline, paired transforms, SegFormer B0/B1 construction, CE/CE+Dice losses, segmentation metrics, `scripts/train_segformer.py`, the PR 01 pytest bootstrap, and the PR 02 frozen inference/eval wrapper. It does **not** yet have semantic-to-cost conversion, planners, route metrics, routing scripts, segmentation sweep tooling, or release-gate tooling.

**Main project goal**: produce an auditable perception-to-routing study:

```text
LoveDA RGB tile
  -> SegFormer semantic prediction
  -> rule-based traversability cost map
  -> grid A* route
  -> route-quality metrics evaluated on GT-semantics-derived oracle cost
```

The project does **not** train a cost-map model in the MVP. LoveDA only provides semantic masks, so both predicted and oracle cost maps are produced by the same deterministic semantic-to-cost policy:

- `predicted_cost = semantic_to_cost(model_prediction)`
- `oracle_cost = semantic_to_cost(gt_semantic_mask)`

This keeps the evaluation interpretable: segmentation errors become route errors, and route safety is measured as crossings through oracle lethal semantic regions such as `building` and `water`.

**Roadmap rules**:

- Each PR does one thing and ships code + pytest.
- Tests use synthetic / hand-computed inputs and should not require LoveDA download, W&B, GPU, or network.
- Routine verification uses `PYTHONPATH=src python -m pytest ...`.
- Real training/routing commands are listed per PR, but long runs are release artifacts, not unit tests.
- The `57.3% mIoU` number is a release gate, not a promise. If no reproducible run reaches `mean_iou >= 0.5730`, report the measured best value.

### PR 01 — Test Bootstrap and Baseline Regression Tests

- **Goal**: Create `tests/`, add `pytest`, and lock basic behavior for the existing codebase.
- **Files**:
  - `tests/__init__.py`
  - `tests/conftest.py`
  - `tests/test_smoke.py`
  - `tests/test_segmentation_core.py`
  - `requirements.txt`
- **Tests**:
  - `import loveda_project` works from tests without manual shell `PYTHONPATH`.
  - `EnsureTensorTypes` normalizes images and masks to expected dtype/range.
  - `MulticlassDiceLoss` ignores class `0`.
  - `SegmentationMeter` ignores class `0` and computes a hand-checked IoU.
  - `build_segformer_model(pretrained=False)` returns an 8-class model.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_smoke.py tests/test_segmentation_core.py -q
```

- **Suggested commit**: `chore: add pytest bootstrap and baseline regression tests`

### PR 02 — Frozen SegFormer Inference Wrapper

- **Goal**: Add a reusable inference API that returns upsampled logits, probabilities, predicted masks, and entropy maps.
- **Files**:
  - `src/loveda_project/inference.py`
  - `scripts/eval_segformer.py`
  - `tests/test_inference.py`
- **Tests**:
  - Random-init B0 + fixed tensor returns logits shaped like the input mask after bilinear upsampling.
  - Per-pixel softmax sums to `1.0`.
  - Entropy is in `[0, log(num_classes)]`.
  - Wrapper runs with `torch.no_grad()` and leaves model train/eval state predictable.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_inference.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/day3_day5_warmcos/checkpoints/best_model.pth \
  --root ./data \
  --output-dir ./outputs/eval_smoke \
  --max-samples 4
```

- **Suggested commit**: `feat: add SegFormer inference wrapper with entropy output`

### PR 03 — Class-Balanced and Focal Losses

- **Goal**: Directly attack the current confusion-matrix failure mode: over-predicted `background` and weak `road` / `barren` / `forest`.
- **Class weighting spec** (`--class-weight-mode`):
  - `none` — disables weighting, equivalent to current behavior.
  - `inverse` — `w_c = N_total / (C * N_c)`; standard inverse-frequency weighting.
  - `effective` — Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019): `w_c = (1 - beta) / (1 - beta^{N_c})` with `beta = 0.9999`, then normalized so `sum_c w_c = C`.
  - `median` — `w_c = median(N) / N_c` (median-frequency balancing).
  - The `ignore` class always receives weight `0`.
- **Class-count source**: a one-time helper `scripts/compute_class_stats.py` (added in PR 03) scans the train split and writes `outputs/class_stats/<train_scenes>.json` containing `{class_name: pixel_count}`. Training reads this JSON via `--class-stats <path>`. The JSON is regenerated whenever `train_scenes` changes (combined / urban-only / rural-only each get their own file).
- **Focal + weights composition**: `loss = focal_term * class_weight[c]`, i.e. focal modulation and class weights multiply per pixel. This matches the standard "weighted focal loss" formulation.
- **Files**:
  - `src/loveda_project/losses.py`
  - `scripts/compute_class_stats.py`
  - `scripts/train_segformer.py`
  - `tests/test_losses_weighted.py`
- **Tests**:
  - Weighted CE with `mode=inverse` on a 3-class hand-built logit/target produces hand-computed weights matching `N_total / (C * N_c)`.
  - `mode=effective` with `beta=0.9999` produces hand-computed weights and sums to `C` after normalization.
  - `ignore` class weight is exactly `0` for all modes.
  - Focal loss with `gamma=2` down-weights easy examples (large-margin correct predictions get smaller loss than hard ones).
  - Focal + `mode=inverse` returns elementwise product of focal term and per-pixel class weight.
  - Existing `ce` and `ce_dice` paths remain unchanged on the same toy logits.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_losses_weighted.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr03_weighted \
  --variant segformer-b0 \
  --pretrained \
  --epochs 35 \
  --batch-size 4 \
  --lr 8e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --loss-name focal \
  --class-weight-mode effective \
  --class-stats outputs/class_stats/urban_rural.json \
  --amp
```

- **Suggested commit**: `feat: add class-balanced and focal segmentation losses`

### PR 04 — Stronger Train-Time Augmentation

- **Goal**: Improve domain robustness with photometric and scale augmentations while preserving image/mask alignment.
- **Preset spec** (`--aug-preset`):
  - `basic` — current behavior: `RandomCropPair(512)`, `RandomHorizontalFlipPair(p=0.5)`, `RandomVerticalFlipPair(p=0.5)`, ImageNet normalize.
  - `strong` — `basic` plus, applied to image only (mask geometry preserved):
    - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)` with `p=0.8`.
    - `RandomScalePair(scale_range=(0.75, 1.5))` applied **before** `RandomCropPair` (image and mask scaled together with bilinear / nearest interpolation respectively).
    - `GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))` with `p=0.3` (image only).
- **Files**:
  - `src/loveda_project/transforms.py`
  - `scripts/train_segformer.py`
  - `tests/test_transforms_aug.py`
- **Tests**:
  - Photometric transforms (`ColorJitter`, `GaussianBlur`) change only the image, not the mask.
  - `RandomScalePair` preserves paired image/mask geometry; image uses bilinear, mask uses nearest.
  - `aug_preset=basic` reproduces current transform behavior bit-exact on a fixed-seed sample.
  - `aug_preset=strong` is deterministic given a seeded RNG (regression-locks the parameter ranges above).
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_transforms_aug.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr04_strong_aug \
  --variant segformer-b0 \
  --pretrained \
  --epochs 35 \
  --batch-size 4 \
  --lr 8e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --aug-preset strong \
  --amp
```

- **Suggested commit**: `feat: add photometric and scale augmentations`

### PR 05 — SegFormer-B2 and Gradient Accumulation

- **Goal**: Add one larger backbone option plus gradient accumulation so the model-capacity ablation is feasible on limited VRAM.
- **Files**:
  - `src/loveda_project/modeling.py`
  - `scripts/train_segformer.py`
  - `tests/test_modeling.py`
  - `tests/test_gradient_accumulation.py`
- **Tests**:
  - B0/B1/B2 all build with `num_labels=8` and `pretrained=False`.
  - Invalid variants raise `ValueError`.
  - Gradient accumulation performs one optimizer step after the configured number of micro-batches.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_modeling.py tests/test_gradient_accumulation.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2 \
  --variant segformer-b2 \
  --pretrained \
  --epochs 35 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --amp
```

- **Suggested commit**: `feat: add SegFormer-B2 and gradient accumulation`

### PR 06 — Sliding-Window and Multi-Scale Evaluation

- **Goal**: Add full-tile / larger-context evaluation for the release metric without changing training.
- **Aggregation rule** (must be stable across release runs):
  - For each window position, run the model and take per-pixel **softmax probabilities** (not raw logits).
  - Accumulate probabilities into a full-tile buffer with a **Gaussian-weighted overlap mask** (sigma = window_size / 4) so window edges contribute less than centers.
  - Normalize the full-tile probability buffer by the accumulated weight per pixel.
  - For multi-scale eval, run the sliding window at each scale, resize the resulting probability map back to the tile resolution (bilinear), and average across scales with equal weight.
  - Final mask is `argmax` over the aggregated probabilities.
- **Files**:
  - `src/loveda_project/inference.py`
  - `scripts/eval_segformer.py` *(extended; created in PR 02)*
  - `tests/test_tta.py`
- **Tests**:
  - A tile smaller than the window matches single-pass inference.
  - Overlapping windows aggregate deterministically with the Gaussian-weighted softmax-average rule (compare against a hand-computed 2-window toy case).
  - Multi-scale eval preserves output size and class dimension.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_tta.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2/checkpoints/best_model.pth \
  --root ./data \
  --output-dir ./outputs/seg_release/pr06_tta_eval \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 0.75 1.0 1.25
```

- **Suggested commit**: `feat: add sliding-window and multi-scale evaluation`

### PR 07 — Config-Driven Segmentation Sweep

- **Goal**: Make the mIoU improvement path auditable by recording comparable runs and a single result table. Also produce the three checkpoints (`combined`, `urban-only`, `rural-only`) needed for the domain-shift matrix.
- **`release.yaml` must define at least these `train_scenes` axes**:
  - `[urban, rural]` — combined training set, used for the release-gate claim.
  - `[urban]` — urban-only checkpoint for the `u→u` and `u→r` matrix cells.
  - `[rural]` — rural-only checkpoint for the `r→r` and `r→u` matrix cells.
- **Files**:
  - `configs/segmentation/release.yaml`
  - `scripts/run_segmentation_sweep.py`
  - `tests/test_segmentation_sweep.py`
- **Tests**:
  - Tiny dry-run config expands combinations deterministically.
  - Sweep produces at least one row per `train_scenes` axis (combined / urban-only / rural-only).
  - Summary table includes `mean_iou`, `road_iou`, `run_dir`, `checkpoint_path`, `train_scenes`, and command.
  - Sweep emits `winner.json` selecting the best `combined` checkpoint by validation `mean_iou`, and copies/symlinks its `best_model.pth` to a stable path `<sweep-dir>/winners/combined/best_model.pth` (and likewise for `urban-only` and `rural-only`).
  - Missing required fields fail with a clear error.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_segmentation_sweep.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/run_segmentation_sweep.py \
  --config configs/segmentation/release.yaml \
  --output-dir outputs/seg_release/sweep
```

- **Suggested commit**: `feat: add config-driven segmentation sweep runner`

### PR 08 — 57.3 mIoU Release Gate

- **Goal**: Enforce the claim gate: only report `57.3% mIoU` when a reproducible metrics JSON actually reaches it.
- **Files**:
  - `scripts/verify_segmentation_release.py`
  - `tests/test_segmentation_release_gate.py`
- **Tests**:
  - Gate passes only when `mean_iou >= 0.5730`.
  - Failing gate prints measured mIoU and the missing delta.
  - Gate verifies `road` IoU exists in `per_class_iou`.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_segmentation_release_gate.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/verify_segmentation_release.py \
  --metrics outputs/seg_release/sweep/best_metrics.json \
  --threshold 0.5730 \
  --require-class road
```

- **Suggested commit**: `feat: add segmentation release gate verifier`

### PR 09 — Rule-Based Semantic-to-Cost Builder

- **Goal**: Convert semantic masks to traversability maps using a transparent YAML policy. This produces both predicted and oracle cost maps.
- **Files**:
  - `src/loveda_project/costmap.py`
  - `configs/costmap/default.yaml`
  - `tests/test_costmap.py`
- **Tests**:
  - Hand-built mask with all LoveDA classes maps to exact configured costs.
  - `building` and `water` become lethal `inf`.
  - `road` becomes the lowest-cost traversable class.
  - Unknown class id raises `ValueError`.
  - `ignore` pixels are filled or marked according to config.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_costmap.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/visualize_costmap.py \
  --mask outputs/eval_smoke/pred_masks/sample_000.png \
  --config configs/costmap/default.yaml \
  --output-dir outputs/costmap_smoke
```

- **Suggested commit**: `feat: add semantic-to-cost map builder`

### PR 10 — Grid A* Planner

- **Goal**: Add the MVP planner used for all Phase-1 perception-to-routing experiments.
- **Files**:
  - `src/loveda_project/planners/__init__.py`
  - `src/loveda_project/planners/astar.py`
  - `tests/test_astar.py`
- **Tests**:
  - Free 10x10 grid returns a shortest path.
  - Wall map routes around `inf` cells.
  - Sealed goal returns `None`.
  - 4-connected and 8-connected costs match hand-computed expectations.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_astar.py -q
```

- **Usage command**: no dataset command needed; this PR is fully covered by synthetic tests.
- **Suggested commit**: `feat: add grid A* planner`

### PR 11 — Valid Start/Goal Sampler

- **Goal**: Sample reproducible route queries on oracle traversable cells so planner failures are attributable to perception/planning, not bad endpoints.
- **Files**:
  - `src/loveda_project/sampling.py`
  - `tests/test_sampling.py`
- **Tests**:
  - Same cost map + same seed returns identical pairs.
  - Every pair starts/ends on finite-cost cells.
  - Every pair satisfies `min_dist`.
  - Optional oracle-A* reachability filter removes disconnected pairs.
  - Fully lethal map returns an empty list quickly.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_sampling.py -q
```

- **Usage command**: no dataset command needed; this PR is fully covered by synthetic tests.
- **Suggested commit**: `feat: add valid start-goal sampler`

### PR 12 — Route-Quality Metrics

- **Goal**: Define how to judge whether a predicted route is good using the oracle cost map derived from GT semantics.
- **Files**:
  - `src/loveda_project/route_metrics.py`
  - `tests/test_route_metrics.py`
- **Tests**:
  - `path_success` is true iff a route exists.
  - Route crossing one lethal cell has expected `lethal_fraction` and `violation_rate`.
  - `excess_cost = cost_oracle(pred_route) / cost_oracle(oracle_route) - 1` matches hand values on a non-violating route.
  - Aggregator over a mixed batch (some violating, some not) returns `excess_cost` computed only over non-violating routes plus `n_valid_routes` equal to that count; never returns `inf`.
  - Missing oracle route produces a clear invalid/evaluation status.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_route_metrics.py -q
```

- **Usage command**: no dataset command needed; this PR is fully covered by synthetic tests.
- **Suggested commit**: `feat: add planning-facing route metrics`

### PR 13 — Phase-1 End-to-End Routing Script

- **Goal**: Run the complete MVP path on a checkpoint: prediction -> cost map -> start/goal -> A* -> route metrics -> JSON/figures.
- **Files**:
  - `scripts/run_phase1_routing.py`
  - `tests/test_phase1_routing_smoke.py`
- **Tests**:
  - Fake one-sample dataset runs end to end.
  - Output JSON contains segmentation metrics, route metrics, config, and sample ids.
  - Figure output is optional in tests but, when enabled, files are non-empty.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_phase1_routing_smoke.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/run_phase1_routing.py \
  --checkpoint outputs/seg_release/sweep/winners/combined/best_model.pth \
  --root ./data \
  --costmap-config configs/costmap/default.yaml \
  --output-dir outputs/phase1_routing/smoke \
  --max-tiles 4 \
  --pairs-per-tile 5
```

- **Suggested commit**: `feat: add Phase 1 routing evaluation script`

### PR 14 — Urban/Rural Domain Routing Matrix

- **Goal**: Evaluate the same checkpoint across `{urban->urban, rural->rural, urban->rural, rural->urban}` settings.
- **Files**:
  - `configs/routing/domain_matrix.yaml`
  - `scripts/run_domain_routing_eval.py`
  - `tests/test_domain_routing_eval.py`
- **Tests**:
  - Tiny fake config expands exactly four domain rows: `(urban-only, urban)`, `(rural-only, rural)`, `(urban-only, rural)`, `(rural-only, urban)`.
  - Each row records train domain, eval domain, checkpoint path (resolved through `winner.json`), cost config, and route metrics.
  - Missing `winner.json` or missing required `train_scenes` axis fails with a clear error.
  - Same seed reproduces the same sampled start/goal pairs.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_domain_routing_eval.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/run_domain_routing_eval.py \
  --config configs/routing/domain_matrix.yaml \
  --output-dir outputs/phase1_routing/domain_matrix
```

- **Suggested commit**: `feat: add urban-rural routing evaluation matrix`

### PR 15 — Uncertainty-Aware Cost Variants

- **Goal**: Add planning-aware use of model confidence without changing the segmentation model: entropy penalty and pessimistic top-2 lethal inflation.
- **Files**:
  - `src/loveda_project/uncertainty.py`
  - `src/loveda_project/costmap.py`
  - `configs/costmap/entropy.yaml`
  - `configs/costmap/pessimistic.yaml`
  - `tests/test_costmap_uncertainty.py`
- **Tests**:
  - Uniform probabilities have entropy `log(C)` (within float tolerance); one-hot probabilities have entropy exactly `0` (no NaN — verifies the `eps` guard).
  - `cost = semantic_cost + lambda * entropy` matches hand values.
  - Road argmax with water/building as high-probability runner-up inflates cost.
  - Non-lethal runner-up does not inflate cost.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_costmap_uncertainty.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/run_phase1_routing.py \
  --checkpoint outputs/seg_release/sweep/winners/combined/best_model.pth \
  --root ./data \
  --costmap-config configs/costmap/entropy.yaml \
  --output-dir outputs/phase1_routing/entropy \
  --max-tiles 20 \
  --pairs-per-tile 10
```

- **Suggested commit**: `feat: add uncertainty-aware traversability costs`

### PR 16 — Perception-to-Routing Benchmark Harness

- **Goal**: Compare checkpoints and cost-map variants in one schema so final claims are backed by artifacts.
- **Files**:
  - `configs/routing/benchmark.yaml`
  - `scripts/run_planner_benchmark.py`
  - `src/loveda_project/benchmark.py`
  - `tests/test_planner_benchmark.py`
- **Tests**:
  - Tiny fake benchmark emits one row per `(checkpoint, costmap, domain)` cell. Checkpoints are resolved by name through `winner.json` (`combined`, `urban-only`, `rural-only`).
  - Output schema includes `mean_iou`, `road_iou`, `route_success`, `excess_cost`, `n_valid_routes`, `lethal_fraction`, `violation_rate`, and `runtime_ms`.
  - Same seed reproduces identical benchmark rows.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_planner_benchmark.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/run_planner_benchmark.py \
  --config configs/routing/benchmark.yaml \
  --output-dir outputs/final_benchmark
```

- **Suggested commit**: `feat: add perception-to-routing benchmark harness`

### PR 17 — Failure Taxonomy and Qualitative Overlays

- **Goal**: Turn route failures into explainable categories and figures for the final report.
- **Files**:
  - `scripts/mine_failure_modes.py`
  - `scripts/make_failure_panels.py`
  - `tests/test_failure_taxonomy.py`
- **Tests**:
  - Fake benchmark JSON with planted examples is categorized into road break, lethal crossing, no route, high excess cost, and uncertainty detour.
  - Panel generation creates non-empty files on synthetic arrays.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_failure_taxonomy.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/mine_failure_modes.py \
  --results outputs/final_benchmark/results.json \
  --output-dir outputs/final_benchmark/failures

PYTHONPATH=src python scripts/make_failure_panels.py \
  --failure-json outputs/final_benchmark/failures/failure_cases.json \
  --output-dir outputs/final_benchmark/panels
```

- **Suggested commit**: `feat: add routing failure taxonomy and overlays`

### PR 18 — Final Artifact Verifier

- **Goal**: Drift-guard for hand-edited claims. The README and report sections are authored manually (not generated by any PR) using numbers from `final_summary.json` produced by PR 16. This verifier compares numbers parsed out of `README.md` / report markdown against `final_summary.json` and fails CI if they disagree, so the hand-edited prose can never silently rot relative to the JSON source of truth.
- **Files**:
  - `scripts/verify_final_artifacts.py`
  - `tests/test_result_artifacts.py`
- **Tests**:
  - Verifier passes when README/report metrics match `final_summary.json`.
  - Verifier fails when `57.3%` is claimed but release gate did not pass.
  - Verifier checks required routing fields for all four domain settings.
- **Pytest command**:

```bash
PYTHONPATH=src python -m pytest tests/test_result_artifacts.py -q
```

- **Usage command**:

```bash
PYTHONPATH=src python scripts/verify_final_artifacts.py \
  --summary outputs/final_benchmark/final_summary.json \
  --readme README.md
```

- **Suggested commit**: `chore: add final artifact consistency verifier`

### PR Status Table / PR 狀態總覽

| PR | Status | Feature | Why it moves the final goal | Test file |
|----|--------|---------|-----------------------------|-----------|
| 01 | ✅ Done | Test bootstrap | Makes all future claims regression-testable | `test_smoke.py`, `test_segmentation_core.py` |
| 02 | ✅ Done | Inference wrapper | Produces logits/probs/entropy for eval and routing | `test_inference.py` |
| 03 | 🔲 Not started | Weighted/focal losses | Targets the 51.2 -> 57.3 mIoU gap | `test_losses_weighted.py` |
| 04 | 🔲 Not started | Strong augmentation | Targets domain shift and robust road/barren prediction | `test_transforms_aug.py` |
| 05 | 🔲 Not started | B2 + grad accumulation | Adds model capacity for the release gate | `test_modeling.py`, `test_gradient_accumulation.py` |
| 06 | 🔲 Not started | Sliding/TTA eval | Improves official eval without retraining | `test_tta.py` |
| 07 | 🔲 Not started | Segmentation sweep | Makes mIoU ablations reproducible | `test_segmentation_sweep.py` |
| 08 | 🔲 Not started | Release gate | Prevents unsupported 57.3 mIoU claims | `test_segmentation_release_gate.py` |
| 09 | 🔲 Not started | Semantic-to-cost | Bridges perception to planning representation | `test_costmap.py` |
| 10 | 🔲 Not started | Grid A* | Adds MVP planner | `test_astar.py` |
| 11 | 🔲 Not started | Start/goal sampler | Creates valid route queries | `test_sampling.py` |
| 12 | 🔲 Not started | Route metrics | Defines route quality | `test_route_metrics.py` |
| 13 | 🔲 Not started | Phase-1 routing script | Runs prediction-to-routing end to end | `test_phase1_routing_smoke.py` |
| 14 | 🔲 Not started | Domain matrix | Measures urban/rural route robustness | `test_domain_routing_eval.py` |
| 15 | 🔲 Not started | Uncertainty cost | Tests confidence-aware routing value | `test_costmap_uncertainty.py` |
| 16 | 🔲 Not started | Benchmark harness | Connects checkpoints to route metrics | `test_planner_benchmark.py` |
| 17 | 🔲 Not started | Failure taxonomy | Explains route failures qualitatively | `test_failure_taxonomy.py` |
| 18 | 🔲 Not started | Artifact verifier | Keeps final claims consistent with JSON | `test_result_artifacts.py` |

### Test Commands Quick Reference / 測試指令速查

```bash
# Full suite
PYTHONPATH=src python -m pytest tests/ -q

# Segmentation release path
PYTHONPATH=src python -m pytest \
  tests/test_segmentation_core.py \
  tests/test_inference.py \
  tests/test_losses_weighted.py \
  tests/test_transforms_aug.py \
  tests/test_modeling.py \
  tests/test_gradient_accumulation.py \
  tests/test_tta.py \
  tests/test_segmentation_sweep.py \
  tests/test_segmentation_release_gate.py -q

# Routing path
PYTHONPATH=src python -m pytest \
  tests/test_costmap.py \
  tests/test_astar.py \
  tests/test_sampling.py \
  tests/test_route_metrics.py \
  tests/test_phase1_routing_smoke.py \
  tests/test_domain_routing_eval.py \
  tests/test_costmap_uncertainty.py \
  tests/test_planner_benchmark.py -q

# Final packaging
PYTHONPATH=src python -m pytest \
  tests/test_failure_taxonomy.py \
  tests/test_result_artifacts.py -q
```

### Post-MVP Extensions / 後續延伸

Hybrid A*, RRT*, cross-view localization, MC-dropout ensembles, and observation-triggered replanning remain good research extensions, but they are not part of the primary MVP roadmap. They should only be scheduled after PR 16 proves that segmentation changes can be evaluated through route-quality metrics.

---

## 9. Writeup Outline / 論文大綱

Final report (4–6 pages + appendix):

1. **Problem Framing** — Why perception-to-planning matters for drone delivery and why mIoU alone is insufficient.
2. **Pipeline Overview** — RGB → semantic prediction → rule-based cost map → grid A* → route metrics.
3. **Current Segmentation Baseline** — Existing SegFormer-B0 result, confusion matrix, and why `road` / `barren` / `forest` are the bottleneck.
4. **Segmentation Release Improvements** — Weighted/focal loss, stronger augmentation, B2 + gradient accumulation, sliding-window/TTA, sweep table, and the `57.3% mIoU` release gate.
5. **Semantic-to-Cost Policy** — Explicitly state that LoveDA has GT semantic masks, not GT cost maps; oracle cost is derived from GT semantics.
6. **Perception-to-Routing Evaluation** — Predicted route planned on predicted cost, then rescored on oracle cost; route success, excess cost, violation rate, lethal fraction, runtime.
7. **Domain Shift and Uncertainty** — Urban/rural matrix and default vs entropy/pessimistic cost policies.
8. **Failure-Mode Taxonomy** — Road breaks, lethal crossings, disconnected routes, high excess cost, and uncertainty detours.
9. **What I'd Build Next** — Learned traversability with real supervision, Hybrid A*/RRT*, localization-coupled planning, temporal imagery, and onboard perception updates.

---

## 10. Risks and Mitigations / 風險與對策

| Risk | Mitigation |
|------|-----------|
| SegFormer-B0 saturates at ~0.51 mIoU | PR 03–08: weighted/focal loss, stronger augmentation, B2, TTA, sweep, release gate |
| Road IoU too low for meaningful routing | Report `road_iou` beside route metrics; add Lovász/crop mining only if PR 03–06 are insufficient |
| Planner bugs confused with perception bugs | Cross-validate grid A* on synthetic maps and oracle cost maps before predicted-cost experiments |
| "GT cost map" wording is challenged | Use precise wording: oracle cost map is derived from GT semantic labels by a deterministic policy |
| Uncertainty metrics don't meaningfully change route quality | Report it honestly; the negative result is still informative |
| Benchmark sweep explodes combinatorially | Pre-declare checkpoint × cost-policy × domain cells in `configs/routing/benchmark.yaml` |
| Tests slow down the loop | Synthetic inputs only; full suite < 60 s target |

---

## 11. Non-Goals / 不做的事情

- Train a foundation VLM from scratch
- Build a simulator or real drone test rig
- Solve SLAM or onboard visual-inertial odometry
- Beat LoveDA SOTA mIoU (we only need **good enough to drive the routing story**)
- Train a direct RGB→cost-map model in the MVP; cost maps are rule-based from semantic labels/predictions
- Claim real-world collision truth; route violations are semantic lethal-region violations
- Build a web frontend; all outputs are JSON + matplotlib figures
- Over-engineer the cost-map YAML schema (the rule-based baseline is a **study subject**, not a product)

---

## 12. Key Dependencies / 關鍵依賴

```
torch>=2.2
transformers>=4.40          # SegFormer
torchgeo>=0.5               # LoveDA loader
numpy>=1.24
matplotlib>=3.7
tqdm
wandb                       # training logging
pytest>=7.0                 # test harness
pyyaml                      # cost-map configs
```

---

## 13. Submission / Milestone Checklist

### Segmentation Release
- [ ] PRs 01–08 merged with green tests
- [ ] Sweep table includes baseline B0, weighted/focal, strong augmentation, B2, and TTA rows
- [ ] Best checkpoint has `best_metrics.json`, run config, confusion matrix, and reproduction command
- [ ] Release gate passes `mean_iou >= 0.5730`, or final wording uses measured best / `targeting 57.3%`
- [ ] `road_iou` is reported beside any headline mIoU

### Perception-to-Routing MVP
- [ ] PRs 09–16 merged with green tests
- [ ] Semantic-to-cost policy is documented as rule-based and GT-semantics-derived for oracle maps
- [ ] `run_phase1_routing.py` produces route JSON and overlays on a real checkpoint
- [ ] Domain matrix reports urban/rural route metrics
- [ ] Benchmark JSON contains `mean_iou`, `road_iou`, route success, excess cost, violation rate, lethal fraction, and runtime

### Final Packaging
- [ ] PRs 17–18 merged with green tests
- [ ] Failure taxonomy panels produced from benchmark JSON
- [ ] README/report headline numbers match generated JSON artifacts
- [ ] No claim says "real collision GT"; wording uses semantic lethal-region violation
