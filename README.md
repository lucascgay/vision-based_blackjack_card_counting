# Vision-Based Blackjack Strategic Assistant

A monorepo with two decoupled components plus shared utilities:

- **`cv_pipeline/`** — Real-time card detection and tracking via YOLOv8, with a Blackjack strategy advisory overlay on video.
- **`RL_Agent/`** — Reinforcement learning agents (TD Q-Learning and Monte Carlo) that learn optimal blackjack play and betting using the Hi-Lo card counting system.
- **`common/`** — Shared data models (card/hand logic) and strategy tables (Illustrious 18 deviations + Basic Strategy).

---

## Setup

Install all dependencies:

```bash
make install
```

Or install each component separately:

```bash
make install-cv    # CV pipeline (YOLOv8, OpenCV, torch, etc.)
make install-rl    # RL agent (gymnasium, numpy, matplotlib)
```

---

## CV Pipeline

The CV pipeline detects and tracks cards in a blackjack video, maintains a running count, and overlays strategy advice onto the output.

### Run the assistant on a video

```bash
make run SOURCE=path/to/video.mp4
```

- Accepts `.mp4`, `.avi`, `.mov` video files
- Annotated output is written to `cv_pipeline/output/annotated_output.mp4`
- Uses pre-trained YOLOv8 weights in `cv_pipeline/detection/weights/`
- Two pre-trained models are included: `table_detector.pt` (detects cards on the table) and `card_classifier.pt` (identifies rank/suit). Both were trained on Google Colab with an H100 GPU.

### Data preparation

1. Download datasets using `cv_pipeline/scripts/download_datasets.sh`.
2. Run dataset preparation:

```bash
python -m cv_pipeline.detection.dataset_prep --config cv_pipeline/config.yaml
```

### Train YOLOv8

```bash
make train-cv
```

Best model is saved to `cv_pipeline/detection/weights/best.pt`.

---

## RL Agent

All RL scripts must be run from the `RL_Agent/` directory:

```bash
cd RL_Agent
```

### 1. Train the agent

Train the Monte Carlo agent (recommended):

```bash
python train_v7_mc.py
```

Or train the TD Q-Learning agent:

```bash
python train_v6.py
```

- Runtime: ~2–4 hours (20M episodes)
- Output: `qtable_v7_mc.npy` or `qtable_v6_final.npy`
- Pre-trained Q-tables are already included in the repo — skip this step to use them.

---

### 2. Evaluate the agent

Compare the trained agent vs Basic Strategy:

```bash
python evaluate.py
```

- Runtime: ~5–10 minutes
- Requires: a trained `.npy` Q-table (auto-detects V7, then V6)
- Output: terminal tables showing win rate, EV/unit, and per-true-count breakdown for flat betting and 1–8 bet spread

---

### 3. Run the bankroll simulation

```bash
python bankroll_sim.py
```

Optional arguments:

```bash
python bankroll_sim.py --sims 1000 --hands 500 --qtable qtable_v7_mc.npy
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--sims` | 1000 | Number of sessions to simulate |
| `--hands` | 500 | Hands per session |
| `--qtable` | auto-detect | Path to Q-table `.npy` file |

- Runtime: ~5–10 minutes
- Output: terminal summary (avg/median bankroll, ruin rate, head-to-head vs Basic Strategy)

---

### 4. Generate figures

```bash
python visualize.py
```

- Runtime: ~1 minute
- Requires: `qtable_v7_mc.npy`
- Output: 7 PNG figures saved to `figures/`

| File | Contents |
|------|----------|
| `fig1_training_curves.png` | Reward convergence over 20M episodes |
| `fig2_ev_by_bucket.png` | EV/unit by true count |
| `fig3_bankroll.png` | Ruin rates and bankroll distribution |
| `fig4_qtable_heatmaps.png` | Action heatmaps (agent vs Basic Strategy) |
| `fig5_agent_vs_bs.png` | Where agent disagrees with Basic Strategy |
| `fig6_qvalue_confidence.png` | Q-value confidence for hard hands |
| `fig7_algorithm_comparison.png` | TD vs Monte Carlo comparison |

---

## Recommended Order

```
train_v7_mc.py  →  evaluate.py  →  bankroll_sim.py  →  visualize.py
```

If you want to skip training, the repo includes pre-trained Q-tables (`qtable_v6_final.npy`, `qtable_v7_mc.npy`) so you can run steps 2–4 directly.
