# Blackjack RL Card Counting Agent

Reinforcement learning agents (TD Q-Learning and Monte Carlo) that learn optimal blackjack play and betting using the Hi-Lo card counting system.

---

## Setup

```bash
pip install gymnasium numpy matplotlib
```

All scripts must be run from the `RL_Agent/` directory:

```bash
cd RL_Agent
```

---

## Running the Scripts

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
