# Latent Dynamics and Efficient Inference in Tiny Recursive Models

**Can a recursive reasoning model stop once its answer has stabilized?** This university research project implements Tiny Recursive Models (TRMs), examines how their hidden states evolve while solving mazes and Sudoku, and tests an acceleration-based stopping rule to reduce inference steps.

At a stopping threshold of **τ = 0.8**, the Maze 10×10 model averaged **7.8 instead of 16 supervision steps** while retaining **96.8% exact-match accuracy** (96.6% at 16 steps). These are reductions in model steps, not measured wall-clock speedups.

## What I built

- Implemented TRMs in PyTorch with both MLP-Mixer and attention variants, including training and evaluation pipelines.
- Tracked the solution state (`y`) and reasoning state (`z`) through recursion; used geometric metrics and PCA to study their trajectories on Maze, Sudoku, and public ARC-AGI checkpoints.
- Added early stopping based on changes in successive `y` updates, with a two-hit check to avoid stopping on a transient signal.
- Tested Matryoshka Representation Learning (MRL) to probe how much of the solution can be recovered from a small prefix of the latent representation.

## Findings

The latent trajectories show an initial coarse-solution phase followed by smaller refinements. The solution state `y` is steadier than the oscillating reasoning state `z`, making it a more useful stopping signal. The report finds unnormalized acceleration more stable in the trajectory plots; the early-stopping implementation in this repository thresholds normalized acceleration.

| Task | Exact match, 16 steps | Exact match, early stopping | Average steps |
| --- | ---: | ---: | ---: |
| Maze 5×5 | 95.2% | 95.0% | 6.4 |
| Maze 10×10 | 96.6% | 96.8% | 7.8 |
| Sudoku 9×9 (attention) | 94.2% | 94.0% | 6.9 |

Early-stopping results use **τ = 0.8** and the models without MRL. Exact match means the entire predicted puzzle solution is correct. MRL generally preserved performance in the reported tasks, with an exception for the Sudoku MLP-Mixer configuration. The experiments were limited by the cost of long recursive runs and the available compute; see the report for the full setup, additional thresholds, plots, and limitations.

## Explore the project

- [Full report](report/main.pdf) · [LaTeX source](report/main.tex)
- [TRM implementation](model/trm.py) · [Training](train.py) · [Evaluation and early stopping](evaluation.py)
- [Dataset preparation](puzzle_datasets/) · [Trajectory analysis](analysis/) · [Recorded results](results/)
- [Experiment notebooks](notebooks/)

The datasets and trained checkpoints are not included in this repository. The dataset preparation scripts and command-line entry points are provided to reproduce the experiments with the required data and model configuration.
