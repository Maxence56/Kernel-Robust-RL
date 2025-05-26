# Kernel-Robust-RL

# Robust MDP Algorithm Comparison Experiments 

## 1. Overview

This script is designed to conduct a comparative analysis of four different algorithms for robust policy evaluation in Markov Decision Processes (MDPs) with transition probability uncertainties. The uncertainties are modeled as L2-norm bounded deviations from a nominal transition kernel.

The primary goal is to evaluate the performance of these algorithms in terms of accuracy (closeness to a benchmark robust value), computational time, and convergence characteristics. The script supports two main types of experiments:

1.  **Single MDP Comparison**: All algorithms are run on the *same* generated MDP instance. This allows for direct comparison of their behavior on an identical problem. Convergence plots are generated for each algorithm, showing the evolution of the robust value estimate per iteration against a benchmark.
2.  **Performance Analysis**: Algorithms are run across multiple (randomly generated) MDP instances. This provides statistical measures of their average performance, including success rates, average computation time, average gap from the benchmark, average number of iterations, and standard deviation of computation time. Box plots summarize these distributions.

The script also includes an older, more general `run_experiments` function that was used for broader comparisons and includes options for Garnet MDPs and JSON result saving. However, the current primary focus for detailed analysis is on `run_single_mdp_comparison` and `run_performance_analysis`.

## 2. Implemented Algorithms

The script implements and compares the following four robust policy evaluation algorithms:

1.  **"Ours (Binary+Spectral)"**:
    *   **Reference**: This method is based on our work "Non-Rectangular Robust MDPs with Normed Uncertainty Sets."
    *   **Method**: This algorithm addresses robust policy evaluation for non-rectangular Lp-norm (specifically L2-norm in the provided spectral method details) uncertainty sets. The core idea is to reframe the problem by finding an optimal Lagrange multiplier (penalty term) λ* associated with the robust Bellman equation. This λ* is found through a binary search. The evaluation step within the binary search, F(λ), involves calculating max<sub>b</sub> ||E<sub>π</sub><sup>λ</sup> b||<sub>q</sub> (for L2, q=2). This maximization is achieved using a spectral method, specifically by computing the largest singular value (or eigenvalue of (E<sub>π</sub><sup>λ</sup>)<sup>T</sup>E<sub>π</sub><sup>λ</sup>) of a specially constructed matrix E<sub>π</sub><sup>λ</sup> that depends on the nominal MDP parameters and λ. The final robust value is then J<sub>nominal</sub> - λ*.
    *   **Uncertainty Model**: Assumes a non-rectangular L2-norm uncertainty set for the transition probabilities.

2.  **"CPI (Frank-Wolfe)"**:
    *   **Reference**: This approach is analogous to Algorithm 3.2 (Conservative Policy Iteration for robust policy evaluation) from "Policy gradient algorithms for robust MDPs with non-rectangular uncertainty sets" by Li, Kuhn, and Sutter (LKS24).
    *   **Method**: It solves the robust policy evaluation problem (max<sub>P ∈ P</sub> V<sup>P</sup><sub>π</sub>) for a fixed policy π under a general convex uncertainty set P for transition probabilities (the script uses an L2 ball). It starts with an initial estimate of the worst-case kernel, P<sup>(m)</sup>. In each iteration, it computes the gradient of the value function V<sup>P<sup>(m)</sup></sup><sub>π</sub> with respect to the transition kernel P. It then solves a linear subproblem to find a direction-giving kernel P<sub>ε</sub> = argmax<sub>P' ∈ P</sub> <∇<sub>P</sub> V<sup>P<sup>(m)</sup></sup><sub>π</sub>, P' - P<sup>(m)</sup>>. This P<sub>ε</sub> is an "extreme" kernel in the uncertainty set that maximizes the current objective improvement. The current kernel estimate is updated via a convex combination: P<sup>(m+1)</sup> = (1 - α<sub>m</sub>)P<sup>(m)</sup> + α<sub>m</sub> P<sub>ε</sub>, where α<sub>m</sub> is a step size. The process continues until the Frank-Wolfe gap (a measure of suboptimality) is below a tolerance.
    *   **Uncertainty Model**: Assumes a general convex uncertainty set for transition probabilities, instantiated as a non-rectangular L2-norm ball in the script.

3.  **"SA-Rectangular"**:
    *   **Reference**: This method employs robust value iteration for (s,a)-rectangular uncertainty sets. The formulation of the robust Bellman operator is consistent with principles described in works like Theorem 3.1 of "Efficient Value Iteration for s-rectangular Robust Markov Decision Processes" by Kumar, Wang, Levy, Mannor (KWLM24), which details such operators for general Lp norms.
    *   **Method**: It is a value iteration algorithm. The robust Bellman operator for (s,a)-rectangular uncertainty modifies the standard Bellman operator by subtracting a penalty term for each state-action pair. This penalty is proportional to β_sa[s,a] * κ_2(V), where β_sa[s,a] is the uncertainty radius for the transition probabilities from state s under action a, and κ_2(V) is the 2-variance (related to standard deviation) of the current value function V. The Bellman update for state s is: V<sub>k+1</sub>(s) = max<sub>a</sub> [ R(s,a) - (reward uncertainty for (s,a)) - γ * β<sub>sa</sub>[s,a] * κ<sub>2</sub>(V<sub>k</sub>) + γ * Σ<sub>s'</sub> P<sub>nominal</sub>(s'|s,a)V<sub>k</sub>(s') ]. The algorithm iteratively applies this operator until the value function converges.
    *   **Uncertainty Model**: Assumes an (s,a)-rectangular uncertainty set. The uncertainty β_sa[s,a] is specific to each state-action pair and independent of others.

4.  **"S-Rectangular "**:
    *   **Reference**: This method is based on robust value iteration for (s)-rectangular uncertainty sets, as detailed in "Efficient Value Iteration for s-rectangular Robust Markov Decision Processes" by Kumar, Wang, Levy, Mannor (KWLM24), particularly Theorem 3.3 and Algorithm 1.
    *   **Method**: It is also a value iteration algorithm. For (s)-rectangular uncertainty, the robust Bellman operator for a state s involves finding an optimal policy (distribution over actions) π_s that maximizes a regularized sum of Q-values: V<sub>k+1</sub>(s) = max<sub>π_s</sub> [ Σ<sub>a</sub> π_s(a)Q<sub>k</sub>(s,a) - (reward uncertainty for s) - γ * β<sub>s</sub>[s] * κ<sub>2</sub>(V<sub>k</sub>) * ||π_s||<sub>2</sub> ], where Q<sub>k</sub>(s,a) = R(s,a) + γ * Σ<sub>s'</sub> P<sub>nominal</sub>(s'|s,a)V<sub>k</sub>(s'), β_s[s] is the uncertainty radius for transitions from state s (shared across actions), and ||π_s||_2 is the of the policy vector for state s. KWLM24 (Algorithm 1) provides an efficient method to solve this inner maximization which involves sorting Q-values and finding a threshold. The algorithm iteratively applies this operator until convergence.
    *   **Uncertainty Model**: Assumes an (s)-rectangular uncertainty set, where the uncertainty radius β_s[s] is defined per state and is shared across all actions originating from that state.

## 3. Script Structure

The script is organized into several key sections:

### 3.1. MDP Utilities
   (`build_P_pi`, `compute_V`, `compute_Q`, `compute_G_based_on_V`, `compute_D_pi`, `compute_d_pi`, `compute_d_sa_pi`, `project_onto_stochastic_matrix`, `project_onto_l2_ball_and_stochastic`)
    *   These are helper functions for common MDP calculations such as building policy-specific transition matrices, computing value functions (V and Q), occupancy measures, and projecting matrices onto valid stochastic or L2-ball constrained spaces.
    *   Numerical stability is addressed by using `scipy.linalg.solve` and `scipy.linalg.inv` where appropriate, with fallbacks to pseudo-inverses (`np.linalg.pinv`) in case of singular matrices.

### 3.2. Algorithm Implementations

*   **Paper 1: Algorithm 1 & 2 (Binary Search + Spectral)**
    *   `spectral_method_F(...)`: Implements Algorithm 2/4 from Paper 1 (Nilim & El Ghaoui, 2005, Appendix G) to compute F(λ). This involves constructing a matrix `A_lambda` and finding the maximum norm `||A_lambda * b\'||_2` subject to constraints on `b\'`, approximated via eigenvalue decomposition of `A_lambda.T @ A_lambda`.
    *   `binary_search_policy_eval_p1(...)`: Implements Algorithm 1. It performs a binary search for λ\* by repeatedly calling `spectral_method_F`.

*   **Paper 2: Algorithm 3.2 (CPI / Frank-Wolfe Adaptation)**
    *   `evaluate_policy_once(...)`: Standard policy evaluation for a fixed kernel P.
    *   `get_discounted_state_visitation(...)`: Computes discounted state visitation frequencies.
    *   `get_adversary_advantage(...)`: Calculates the advantage function used by the adversary.
    *   `cpi_policy_eval_p2(...)`: The main loop for the adapted CPI algorithm. It iteratively finds the worst-case kernel `P_epsilon` and updates the current kernel `P_current`.
    *   The original `find_direction_kernel_L2` function is an older component related to a more direct Frank-Wolfe on the kernel space, while `cpi_policy_eval_p2` uses a structure more aligned with adversarial training.

*   **Rectangular Models (Value Iteration)**
    *   `kappa_2_variance(...)`: Calculates the `kappa_2(v)` term used in the rectangular robust Bellman operators.
    *   `policy_evaluation_sa_rectangular_L2(...)`: Implements one step of the robust Bellman backup for (s,a)-rectangular.
    *   `robust_value_iteration_sa_rect_L2(...)`: Wrapper that iteratively calls `policy_evaluation_sa_rectangular_L2` until convergence for the SA-Rectangular model.
    *   `policy_evaluation_s_rectangular_L2(...)`: Implements one step of the robust Bellman backup for (s)-rectangular uncertainty.
    *   `robust_value_iteration_s_rect_L2(...)`: Wrapper that iteratively calls `policy_evaluation_s_rectangular_L2` until convergence for the S-Rectangular model.

### 3.3. Experiment Setup
    (`generate_random_mdp`, `generate_garnet_mdp`, `sample_kernel_from_L2_ball`)
    *   `generate_random_mdp(...)`: Creates a random MDP with S states and A actions. Transition probabilities and rewards are drawn from uniform distributions.
    *   `generate_garnet_mdp(...)`: Creates a Garnet MDP, characterized by a branching factor `b_param` determining the number of successor states for each state-action pair.
    *   `sample_kernel_from_L2_ball(...)`: Samples a transition kernel `P_sample` from (approximately) the boundary of an L2 ball of radius `beta` centered at `P_nominal`. This is used for benchmarking by estimating the true worst-case value.

### 3.4. Experiment Execution Functions

*   **`run_single_mdp_comparison(...)`**:
    *   Generates a single MDP instance (either random or Garnet).
    *   Computes a benchmark robust value by sampling many kernels from the L2 ball around the nominal kernel and finding the minimum expected value.
    *   Runs all four algorithms on this MDP.
    *   Reports the computed robust value, execution time, gap from the benchmark minimum, and number of iterations for each algorithm.
    *   Optionally plots the convergence trajectories of the algorithms. The plot shows the estimated robust value at each iteration, the benchmark minimum line, and special markers for convergence points. Lines are extended post-convergence to the maximum iteration count among all algorithms for better visual comparison.

*   **`run_performance_analysis(...)`**:
    *   Conducts `n_trials` experiments. In each trial:
        *   A new MDP instance and a new random policy are generated.
        *   A benchmark robust value is computed (using fewer samples than `run_single_mdp_comparison` for speed).
        *   Each of the four algorithms is run.
    *   Aggregates results across trials: success rate, average time, average gap from benchmark, average iterations, and standard deviation of time.
    *   Generates box plots comparing the distributions of computation time, gaps, and iterations across algorithms.
    *   Generates a bar chart for success rates.

*   **`run_experiments(...)` (Legacy)**:
    *   This is an older, more general-purpose experiment function. It was initially designed for comparing primarily the "Ours (Binary+Spectral)" and "CPI (Frank-Wolfe)" algorithms, along with the rectangular models.
    *   It supports multiple trials, generation of random or Garnet MDPs.
    *   It calculates a benchmark robust value by sampling.
    *   It collects various metrics (time, value, gap, iterations) for each algorithm across trials.
    *   It saves the aggregated results to a timestamped JSON file.
    *   It includes several plotting options:
        *   `plot_single_trial_convergence`: Plots convergence for the first successful trial.
        *   `plot_aggregated_convergence`: Plots mean convergence trajectories with error bands (±1 standard deviation) across all trials.
        *   `plot_time_comparison`: Box plot of computation times.
        *   `plot_gap_comparison`: Box plot of suboptimality gaps.
    *   The rectangular model parameters (`beta_sa_mat_trial`, `beta_s_vec_trial`) are currently hardcoded within this function for simplicity in its legacy state.

### 3.5. Main Execution Block (`if __name__ == "__main__":`)
    *   Sets common parameters for the experiments (e.g., tolerances, max iterations, gamma).
    *   Provides an `RUN_PERFORMANCE_ANALYSIS` flag to enable/disable the multi-trial performance analysis section.
    *   Defines and runs specific sets of experiments for `run_single_mdp_comparison`:
        *   **Set 1**: Varying state space size S = [10, 50, 100, 200] with fixed A=10, β=0.01.
        *   **Set 2**: Varying uncertainty radius β = [0.005, 0.01, 0.05, 0.1] with fixed S=100, A=10.
    *   If `RUN_PERFORMANCE_ANALYSIS` is true, runs sets of experiments for `run_performance_analysis`:
        *   **Analysis 1**: Varying state space size S = [50, 100, 150, 200].
        *   **Analysis 2**: Varying uncertainty radius β = [0.005, 0.01, 0.02, 0.05].
    *   Prints summary messages about the completed experiments.

## 4. Dependencies

*   **NumPy**: For numerical operations, especially array manipulations.
*   **SciPy**: For linear algebra functions (e.g., `scipy.linalg.solve`, `scipy.linalg.eigh`, `scipy.linalg.inv`).
*   **Matplotlib**: For generating plots.
*   **tqdm** (Optional): For displaying progress bars during long computations (e.g., benchmark sampling).
*   **json**: For saving results (used in the legacy `run_experiments` function).
*   **datetime**: For timestamping saved result files (used in `run_experiments`).

## 5. How to Run

1.  Ensure all dependencies are installed.
2.  Execute the script from the command line:
    ```bash
    python experimentsv2.py
    ```
3.  The script will print output to the console, including parameters for each run, results from each algorithm, and summary statistics.
4.  Matplotlib plots will be displayed interactively.
5.  To enable the multi-trial performance analysis (which can be time-consuming), set the `RUN_PERFORMANCE_ANALYSIS` flag to `True` in the main execution block of the script.

## 6. Key Parameters and Settings

### 6.1. Common Parameters (in `__main__`):
    *   `N_SAMPLES_OPT_single`: Number of samples for benchmark calculation in `run_single_mdp_comparison` (default: 1000). Higher values yield more accurate benchmarks but take longer.
    *   `N_SAMPLES_OPT_multi`: Number of samples for benchmark calculation in `run_performance_analysis` (default: 100). Lower for faster multi-trial runs.
    *   `P1_TOL_common`: Tolerance for the binary search in "Ours (Binary+Spectral)" (default: 1e-6).
    *   `P2_TOL_common`: Tolerance for the Frank-Wolfe gap in "CPI (Frank-Wolfe)" and for value iteration in rectangular models (default: 1e-6).
    *   `CPI_MAX_ITER_common`: Maximum iterations for "CPI (Frank-Wolfe)" and value iteration in rectangular models (default: 100).
    *   `GAMMA_common`: Discount factor γ for the MDPs (default: 0.9).
    *   `N_TRIALS_performance`: Number of trials for `run_performance_analysis` (default: 20).

### 6.2. Experiment Control Flags (in `__main__`):
    *   `RUN_PERFORMANCE_ANALYSIS`: Boolean flag to enable or disable the execution of `run_performance_analysis` experiments (default: `False`).

### 6.3. Algorithm-Specific Parameters:
    *   **`run_single_mdp_comparison` / `run_performance_analysis`**:
        *   `S`: Number of states.
        *   `A`: Number of actions.
        *   `beta`: L2-norm uncertainty radius β.
        *   `gamma`: Discount factor.
        *   `mdp_type`: "random" or "garnet".
        *   `garnet_b_param`: Branching factor for Garnet MDPs (if `mdp_type="garnet"`).
        *   `plot_convergence` (for `run_single_mdp_comparison`): Boolean to enable/disable convergence plotting.
    *   The rectangular model implementations (`robust_value_iteration_sa_rect_L2`, `robust_value_iteration_s_rect_L2`) currently use the global `beta` parameter to define their `beta_sa_mat` (as `beta * np.ones((S,A))`) and `beta_s_vec` (as `beta * np.ones(S)`). This implies that for these experiments, the `beta` for the non-rectangular models is interpreted as a uniform per-state-action or per-state budget for the rectangular models.

## 7. Output and Interpretation

### 7.1. Console Output:
    *   Experiment parameters (S, A, β, etc.).
    *   For each algorithm:
        *   Computed robust value.
        *   Wall-clock execution time.
        *   Gap from the benchmark minimum (Robust Value_algo - Robust Value_benchmark). A smaller, non-negative gap is better.
        *   Number of iterations to convergence.
        *   Status (e.g., "✅ Conv", "⚠️ Max", "❌ Error").
    *   Summary tables for direct comparison.
    *   For performance analysis: aggregated statistics (mean, std. dev., success rate).

### 7.2. Plots:
    *   **Single MDP Convergence Plot**:
        *   X-axis: Iteration number.
        *   Y-axis: Estimated robust value.
        *   Lines for each algorithm show how their value estimate evolves.
        *   A horizontal line indicates the benchmark minimum robust value.
        *   Star markers indicate the point of convergence for each algorithm.
        *   Dotted lines extend post-convergence to allow comparison of convergence speed.
    *   **Performance Analysis Box Plots**:
        *   **Computation Time**: Distribution of wall-clock times for each algorithm. Lower is better.
        *   **Accuracy (Gap)**: Distribution of gaps from the benchmark minimum. Lower (closer to zero) is better.
        *   **Convergence Speed (Iterations)**: Distribution of iterations to convergence. Lower is better.
    *   **Performance Analysis Bar Chart**:
        *   **Reliability (Success Rate %)**: Percentage of trials where each algorithm successfully converged. Higher is better.

### 7.3. JSON Output (Legacy `run_experiments` function):
    *   If the `run_experiments` function is used, it saves detailed results (parameters, times, values, gaps, iterations, convergence data lists, benchmark values) to a timestamped JSON file (e.g., `experiment_results_random_S10_A5_beta0.1_20231027_153000.json`). This allows for later re-analysis or plotting.

## 8. Notes for Reviewers

*   The script provides a framework for comparing different classes of robust policy evaluation algorithms: those for general L2 uncertainty and those for more structured rectangular uncertainties (value iteration based).
*   The benchmark robust value is estimated via sampling, which is an approximation. The quality of this approximation depends on `N_SAMPLES_OPT_single` or `N_SAMPLES_OPT_multi`.
*   The Frank-Wolfe based algorithm (`cpi_policy_eval_p2`) is an adaptation. Its step-size rule and specific formulation details are particular to this implementation for the L2 robust policy evaluation task.
*   The current setup for rectangular models in the main experiment loops (`run_single_mdp_comparison`, `run_performance_analysis`) directly uses the global `beta` parameter. This means if `beta = 0.01` is set for a non-rectangular comparison, the SA-rectangular model will use `beta_sa[s,a] = 0.01` for all (s,a), and the S-rectangular model will use `beta_s[s] = 0.01` for all `s`.
*   The script handles potential numerical issues in MDP calculations (e.g., singular matrices) by falling back to pseudo-inverses, which can affect precision but improves robustness of the script.
*   The plotting features are designed to give a clear visual comparison of algorithm performance, both on individual problems and statistically across many problems.
