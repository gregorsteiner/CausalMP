## Inference on counterfactual distributions using martingale posteriors

This repository contains the code to reproduce all results in the paper "Inference on counterfactual distributions using martingale posteriors" ([arXiv:2607.24143](https://arxiv.org/abs/2607.24143)).
The methods are implemented in `causal_mp.py` (unconfoundedness) and `causal_mp_iv.py` (instrumental variables). The code in the `pr_copula` folder is a slight modification of the pr_copula package available at https://github.com/edfong/MP.

### Setup

The results were produced with Python 3.14. Install the dependencies with

```bash
pip install -r requirements.txt
```

The benchmark in `simulated_examples/illustration.py` additionally requires R with the packages `ranger` and [npcausal](https://github.com/ehkennedy/npcausal) (`remotes::install_github("ehkennedy/npcausal")`). If `rpy2` cannot find R, set the `R_HOME` environment variable.

### Reproducing the results

From the repository root, run

```bash
python run_all.py
```

Add `--skip-r` to skip the R benchmark and `--skip-slow` to skip the (long) coverage experiment. Each example can also be run on its own from the repository root, e.g.

```bash
python -m real_examples.zinc
```

The real-data examples are in `real_examples/` and the simulations in `simulated_examples/`. Every example writes its figure to `Figures/` and prints the numbers reported in the paper.
