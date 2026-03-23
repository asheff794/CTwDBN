# CTwDBN (based on HMDBN)
Continuous-Time weighted Dynamic Bayesian Network (CTwDBN) — an approach for estimating time-evolving, weighted directed graphs — implemented on top of the original HMDBN codebase for computational speed.

## What is this?
CTwDBN extends non-stationary DBNs to output a continuously evolving, weighted DAG at every time point. It works for both quasi-stationary regimes and smoothly varying dependency trajectories. We use HMDBN as the inference engine (for speed and robustness) and add:
- Continuous DAG construction (time-resolved edge probabilities)
- Optional zero-phase Gaussian smoothing (FFT-based)
- End-to-end tutorial scripts for two synthetic settings

### CTwDBN pipeline (high level)
1) Fit HMDBN models
	- HMDBN (Zhu & Wang) nests a DBN inside each hidden state of an HMM so the directed structure can change over time.
	- Structure learning uses a node-wise structural EM routine with BWBIC for model selection.
2) Build time-evolving graphs
	- For each target j, marginalize over graph states to obtain the edge probability P(i→j, t) as the sum of posteriors for all states that include i as a parent of j.
	- This preserves probabilistic information at every time point (no thresholded crossings).
3) Aggregate and smooth
	- Average across multiple seeds/trials to obtain a single robust time-evolving graph.
	- Optionally apply a zero-phase Gaussian smoother (FFT) along time.
4) Analyze
	- Vectorize edges over time and run PCA for visualization/trajectories.
	- K-means clustering with bootstrap to identify discrete states (inertia + silhouette).

## Dependencies
- MATLAB (tested with Parallel Computing Toolbox for parfor)

## Tutorials (start here)
Two self-contained MATLAB scripts demonstrate CTwDBN end-to-end. Run from the repository root so `Code/` and `Data/` are present.

- `Code/CTwDBN/Code/CTwDBN_Trial_Example.m`
  - Trial-based tutorial (continuous dependency trajectory, Bernoulli data).
  - Generates many trials, fits CTwDBN per trial, averages continuous DAGs, and compares to ground truth.

- `Code/CTwDBN/Code/CTwDBN_Spontaneous_Example.m`
  - Spontaneous tutorial (quasi-stationary, piecewise-constant Poisson data).
  - Consolidates E/I to 3 populations, fits CTwDBN across seeds, averages then smooths, and analyzes with bootstrap K-means and PCA.

Core learning entry point:
- `Code/CTwDBN/Code/ctwdbn_structEM.m` — CTwDBN structure learning and continuous DAG output (based on HMDBN).

Data generators and helpers:
- `Code/CTwDBN/Code/simulate_struct_bernoulli_bin_3d.m` — Trial-based Bernoulli generator (supports multi-trial mode).
- `Code/CTwDBN/Code/simulate_poisson_struct_session.m` — Spontaneous Poisson session with hard-coded state-dependent connectivity.
- `Code/fft_gauss_reflect.m` — Fast zero-phase Gaussian smoothing via FFT with reflective padding.

## Citation and Acknowledgements
- CTwDBN by Alec Sheffield
	- Sheffield, Alec G., Sachira Denagamage, Mitchell P. Morton, Anirvan S. Nandy, and Monika P. Jadi. 2026. “Uncovering Dynamic Neural Information Flow with Continuous-Time Weighted Dynamic Bayesian Networks.” bioRxivorg. bioRxiv. https://doi.org/10.64898/2026.01.22.701045.

- Built atop HMDBN
	- Shijia Zhu & Yadong Wang, Hidden Markov induced Dynamic Bayesian Network for recovering time evolving gene regulatory networks, Scientific Reports 5:17841 (2015). [(link)](https://www.nature.com/articles/srep17841.pdf)