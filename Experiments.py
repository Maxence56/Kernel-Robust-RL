import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from tqdm import tqdm # Optional: for progress bars
import json             # <-- For saving results
import datetime         # <-- For timestamp in filename

# #############################################
# ####### MDP Utilities              ##########
# #############################################

def build_P_pi(P, pi, S, A):
    """Builds the policy-specific transition matrix P_pi."""
    P_pi = np.zeros((S, S))
    for s in range(S):
        # Vectorized version
        P_pi[s, :] = np.sum(pi[s, :, np.newaxis] * P[s, :, :], axis=0)
        # Original loop version (equivalent)
        # for a in range(A):
        #     P_pi[s, :] += pi[s, a] * P[s, a, :]
    return P_pi

def compute_V(P_pi, R_pi, gamma, S):
    """Computes the value function V^pi for a given P_pi and R_pi."""
    try:
        I = np.identity(S)
        # Use scipy's solve for potentially better stability/performance
        V = scipy.linalg.solve(I - gamma * P_pi, R_pi, assume_a='pos')
        return V
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix encountered in compute_V. Using pseudo-inverse.")
        # Fallback to pseudo-inverse
        return np.linalg.pinv(I - gamma * P_pi) @ R_pi
    except ValueError as e:
         print(f"Warning: scipy.linalg.solve failed ({e}). Using pseudo-inverse.")
         I = np.identity(S) # Redefine I just in case
         return np.linalg.pinv(I - gamma * P_pi) @ R_pi


def compute_Q(V, R, P, gamma, S, A):
    """Computes the action-value function Q^pi."""
    Q = np.zeros((S, A))
    # Vectorized computation
    Q = R + gamma * np.einsum('sat,t->sa', P, V)
    # Original loop version (equivalent)
    # for s in range(S):
    #     for a in range(A):
    #         Q[s, a] = R[s, a] + gamma * P[s, a, :] @ V
    return Q

def compute_G_based_on_V(V, R, gamma, S, A):
    """Computes G^pi(s,a,s') = R(s,a) + gamma * V^pi(s')"""
    G = np.zeros((S, A, S))
    # Vectorized
    G = R[:, :, np.newaxis] + gamma * V[np.newaxis, np.newaxis, :]
    # Original loop version (equivalent)
    # for s in range(S):
    #     for a in range(A):
    #         for s_prime in range(S):
    #             G[s, a, s_prime] = R[s, a] + gamma * V[s_prime]
    return G

def compute_D_pi(P_pi, gamma, S):
    """Computes the occupancy matrix D^pi = (I - gamma * P_pi)^-1."""
    I = np.identity(S)
    try:
        # Use scipy's inv for potentially better stability/performance
        D = scipy.linalg.inv(I - gamma * P_pi)
        return D
    except np.linalg.LinAlgError:
         print("Warning: Singular matrix encountered in compute_D_pi. Using pseudo-inverse.")
         # Fallback to pseudo-inverse
         return np.linalg.pinv(I - gamma * P_pi)
    except ValueError as e:
         print(f"Warning: scipy.linalg.inv failed ({e}). Using pseudo-inverse.")
         I = np.identity(S) # Redefine I just in case
         return np.linalg.pinv(I - gamma * P_pi)


def compute_d_pi(D_pi, mu, S):
    """Computes the state occupancy measure d^pi."""
    # d(s) = sum_{s0} mu(s0) * D_pi(s0, s)
    # This is equivalent to mu^T @ D_pi
    d = mu @ D_pi
    return d

def compute_d_sa_pi(d_pi, pi, S, A):
    """Computes the state-action occupancy measure d_sa^pi."""
    # d_sa(s,a) = d_pi(s) * pi(a|s)
    d_sa = d_pi[:, np.newaxis] * pi
    return d_sa

# Helper to ensure policy is accessible
CURRENT_POLICY = None
def set_policy(pi):
    global CURRENT_POLICY
    CURRENT_POLICY = pi
def get_policy():
    return CURRENT_POLICY


def project_onto_stochastic_matrix(P):
    """Projects a matrix P (S, A, S) onto the set of stochastic matrices."""
    S, A, _ = P.shape
    # 1. Ensure non-negativity
    P_clipped = np.maximum(P, 0)

    # 2. Ensure rows sum to 1
    row_sums = P_clipped.sum(axis=2, keepdims=True)
    # Avoid division by zero for rows that became all zero
    row_sums[row_sums < 1e-12] = 1.0 # If sum is very small, treat as uniform? Or distribute error? Let's make it sum to 1.

    P_stochastic = P_clipped / row_sums

    # Final check due to potential floating point issues if needed
    # final_row_sums = P_stochastic.sum(axis=2)
    # if not np.allclose(final_row_sums, 1.0):
    #     print("Warning: Row sums not exactly 1 after projection. Re-normalizing.")
    #     P_stochastic = P_stochastic / P_stochastic.sum(axis=2, keepdims=True)

    return P_stochastic


def project_onto_l2_ball_and_stochastic(P_target, P_center, beta):
    """Projects P_target onto the L2 ball around P_center AND ensures stochasticity."""
    S, A, _ = P_center.shape
    delta = P_target - P_center
    norm_delta = np.linalg.norm(delta) # Frobenius norm

    # 1. Project onto L2 ball
    if norm_delta > beta + 1e-9: # Add tolerance for floating point
        delta_proj = delta * (beta / norm_delta)
    else:
        delta_proj = delta

    P_in_ball = P_center + delta_proj

    # 2. Project onto stochastic matrices
    P_final = project_onto_stochastic_matrix(P_in_ball)

    # 3. (Optional but recommended) Check if the projection moved it outside the ball
    final_delta = P_final - P_center
    final_norm = np.linalg.norm(final_delta)
    if final_norm > beta + 1e-7: # Allow slightly larger tolerance
        # print(f"Warning: Stochastic projection moved point outside L2 ball ({final_norm:.4f} > {beta:.4f}). Re-projecting delta.")
        # Re-project the delta needed to reach the stochastic version
        P_final = P_center + final_delta * (beta / final_norm)
        # And ensure stochasticity again
        P_final = project_onto_stochastic_matrix(P_final)


    return P_final

# #############################################
# ####### Paper 1: Algorithm 1 & 2   ##########
# #############################################

def spectral_method_F(lam, beta, pi, P_nominal, R, gamma, mu, S, A):
    """
    Computes F(lambda) using the spectral method (Algorithm 2 / 4 interpretation from Paper 1 Appendix G).
    F(lambda) = beta * max_{||b'||_2 <= 1, b'>=0} ||A_lambda * b'||_2
    """
    # --- Construct the matrix A_lambda (S x SA) ---
    P_pi_nom = build_P_pi(P_nominal, pi, S, A)
    R_pi_nom = np.sum(pi * R, axis=1)
    try:
        V_nom = compute_V(P_pi_nom, R_pi_nom, gamma, S)
        D_pi_nom = compute_D_pi(P_pi_nom, gamma, S) # S x S
    except Exception as e:
        print(f"Error computing nominal V or D_pi: {e}. Returning F(lambda)=0")
        return 0.0

    d_pi_nom = compute_d_pi(D_pi_nom, mu, S) # S

    # Term is v^pi d^pi^T (interpreted as outer product)
    term1 = D_pi_nom @ np.outer(V_nom, d_pi_nom) # S x S

    term2 = lam * D_pi_nom # S x S

    # Projection matrix (onto space orthogonal to all-ones vector)
    projection_matrix = np.identity(S) - (1.0 / S) * np.ones((S, S)) # S x S

    # Matrix representing the linear operator within the norm calculation
    # A_lambda(b) = gamma * projection_matrix @ (term1 - term2) @ H_pi(b)
    # We compute the matrix explicitly
    A_lambda = np.zeros((S, S * A))
    H_pi_term = (term1 - term2) # S x S

    # Precompute H_pi(basis_vectors) implicitly
    # H_pi(b) results in a vector h_s = sum_a pi(a|s)b(s,a)
    # A_lambda[r, c] where c corresponds to (s,a)
    col_idx = 0
    for s_in in range(S):
        for a_in in range(A):
            # Effect of basis vector e_{s_in, a_in} through H_pi
            h_pi_basis_effect = np.zeros(S)
            h_pi_basis_effect[s_in] = pi[s_in, a_in] # Only state s_in is affected

            # Apply the D_pi related part
            intermediate_vec = H_pi_term @ h_pi_basis_effect # S

            # Apply projection
            projected_vec = projection_matrix @ intermediate_vec # S

            A_lambda[:, col_idx] = gamma * projected_vec
            col_idx += 1

    # --- Now use Algorithm 4 (Second Order Spectral Approx) ---
    matrix_AtA = A_lambda.T @ A_lambda # (S*A) x (S*A)

    try:
        # Use eigh for symmetric matrices
        eigenvalues, eigenvectors = scipy.linalg.eigh(matrix_AtA)
        # eigh returns eigenvalues in ascending order, so reverse them
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        # Clip tiny negative eigenvalues that might arise from numerical issues
        eigenvalues = np.maximum(eigenvalues, 0)
    except np.linalg.LinAlgError:
        print("Warning: Eigenvalue computation failed. Returning F(lambda)=0.")
        return 0.0

    best_approx_val_sq = 0.0

    # Check top eigenvectors (adjust num_vecs_to_check based on performance needs)
    num_vecs_to_check = min(S*A, 10) # Heuristic limit

    for i in range(num_vecs_to_check):
        if eigenvalues[i] < 1e-12: # Ignore negligible eigenvalues
            continue

        vi = eigenvectors[:, i]
        vi_plus = np.maximum(vi, 0)
        norm_vi_plus = np.linalg.norm(vi_plus)

        if norm_vi_plus < 1e-12:
            continue

        ui = vi_plus / norm_vi_plus # Normalized positive part

        # Compute || A_lambda @ ui ||^2
        Ax = A_lambda @ ui
        approx_val_sq = np.dot(Ax, Ax)

        if approx_val_sq > best_approx_val_sq:
             best_approx_val_sq = approx_val_sq

    max_norm = np.sqrt(best_approx_val_sq)

    return beta * max_norm


def binary_search_policy_eval_p1(pi, P_nominal, R, gamma, mu, beta, S, A, tolerance=1e-7, max_iter=100):
    """Implements Algorithm 1 from Paper 1."""
    set_policy(pi) # Make policy globally accessible for utils if needed

    # Compute nominal value once
    P_pi_nom = build_P_pi(P_nominal, pi, S, A)
    R_pi_nom = np.sum(pi*R, axis=1)
    try:
      V_nom = compute_V(P_pi_nom, R_pi_nom, gamma, S)
      J_nominal = V_nom @ mu
    except Exception as e:
        print(f"Error computing nominal value: {e}. Aborting P1 eval.")
        return np.nan, 0, {}


    # Robust estimate of upper bound for lambda* (penalty)
    # Max possible change in value is roughly beta * ||gradient|| * some_factor
    # Let's use a simple reward-based bound if rewards are scaled, e.g. to [0, 1]
    Rmax = np.max(np.abs(R)) if R.size > 0 else 1.0
    # lambda_high = beta * Rmax / (1 - gamma)**2 # Heuristic, might be too loose/tight
    lambda_high = Rmax / (1 - gamma) # Upper bound on total value range

    lambda_low = 0.0
    lambda_mid = 0.0
    n_iter = 0

    iterations = []
    lambdas = []
    values = []
    f_lambdas = []

    start_time = time.time()
    for i in range(max_iter):
        n_iter = i + 1
        lambda_mid = (lambda_low + lambda_high) / 2.0
        
        # Check for convergence of lambda itself
        if (lambda_high - lambda_low) < tolerance:
             # print(f"P1 converged in {i} iterations (lambda range)")
             break

        f_lambda = spectral_method_F(lambda_mid, beta, pi, P_nominal, R, gamma, mu, S, A)

        iterations.append(i)
        lambdas.append(lambda_mid)
        values.append(J_nominal - lambda_mid)
        f_lambdas.append(f_lambda)

        # F(lambda) should be decreasing in lambda based on definition? Let's test.
        # If f(lambda) > lambda, it means the penalty calculated using lambda_mid
        # is larger than lambda_mid itself. The true fixed point lambda* must be larger.
        if f_lambda > lambda_mid:
            lambda_low = lambda_mid
        else:
            # f(lambda) <= lambda, fixed point is less than or equal to lambda_mid
            lambda_high = lambda_mid

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Use the upper bound of the final interval as the best estimate for lambda*
    robust_value = J_nominal - lambda_high
    # print(f"P1 finished: lambda* estimate = {lambda_high:.7f}")

    convergence_data = {'iterations': iterations, 'lambdas': lambdas, 'values': values, 'f_lambdas': f_lambdas}

    return robust_value, elapsed_time, convergence_data, n_iter

# #############################################
# ####### Paper 2: Algorithm 3.2     ##########
# #############################################

def find_direction_kernel_L2(P_current, P_nominal, beta, pi, R, gamma, mu, S, A):
    """Finds the kernel P_e maximizing the expression <W, P_e> for L2 ball."""
    # Calculate required components for P_current
    P_pi_curr = build_P_pi(P_current, pi, S, A)
    R_pi_curr = np.sum(pi * R, axis=1)
    try:
        V_curr = compute_V(P_pi_curr, R_pi_curr, gamma, S)
        Q_curr = compute_Q(V_curr, R, P_current, gamma, S, A)
        # Use V_curr to compute G
        G_curr = compute_G_based_on_V(V_curr, R, gamma, S, A)
        D_pi_curr = compute_D_pi(P_pi_curr, gamma, S)
        d_pi_curr = compute_d_pi(D_pi_curr, mu, S)
    except Exception as e:
         print(f"Error computing values in direction finding: {e}. Returning P_nominal.")
         # Return nominal and a large gap to indicate failure? Or zero gap? Let's return zero gap.
         return P_nominal, 0.0 # Return nominal and zero gap


    # Calculate Advantage A(s,a,s') = G(s,a,s') - V(s) ? No, G(s,a,s') - Q(s,a)
    # Let's stick to the paper's gradient definition W(s,a,s') = (1/(1-gamma)) * d_pi_curr(s) * pi(a|s) * A_adv(s,a,s')
    # Where A_adv is the advantage G(s,a,s') - Q(s,a)

    # Calculate Advantage A_adv(s,a,s') = G(s,a,s') - Q(s,a)
    A_adv = np.zeros((S, A, S))
    # Vectorized A_adv(s,a,s') = G(s,a,s') - Q(s,a)
    A_adv = G_curr - Q_curr[:, :, np.newaxis]


    # Calculate gradient W(s,a,s') = (1/(1-gamma)) * d_pi_curr(s) * pi(a|s) * A_adv(s,a,s')
    W = np.zeros((S, A, S))
    d_pi_expanded = d_pi_curr[:, np.newaxis, np.newaxis] # Expand d_pi for broadcasting
    pi_expanded = pi[:, :, np.newaxis] # Expand pi for broadcasting

    # Need to be careful with the definition of W and the Frank-Wolfe objective.
    # Standard FW aims to minimize f(x) by finding x* = argmin <grad f(x_k), x>.
    # Here, we want to *minimize* J^pi(P) = V^pi(P) @ mu over P in the uncertainty set U.
    # The gradient of J^pi(P) w.r.t P (at P=P_current) has components related to W.
    # grad_P J(P_current) [s,a,s'] \propto d^pi_curr(s) * pi(a|s) * A_adv(s,a,s')
    # So W is proportional to the gradient.
    # Minimizing J means we want to find P_e that *minimizes* <grad, P_e>.
    # P_e = argmin_{P in U} <W, P>

    W = (1.0 / (1.0 - gamma)) * d_pi_expanded * pi_expanded * A_adv

    # We want to find P_e in the L2 ball around P_nominal that minimizes <W, P_e>
    # Minimize <W, P_nominal + Delta_e> subject to ||Delta_e||_F <= beta, sum_{s'} Delta_e = 0
    # Minimize <W, Delta_e> subject to constraints.
    # Solution Delta_e is proportional to -W projected onto sum-zero constraint.

    # Project W onto row-sum-zero subspace (for Delta):
    row_sums_W = W.sum(axis=2, keepdims=True)
    W_proj = W - row_sums_W / S # Subtract the mean

    norm_W_proj = np.linalg.norm(W_proj)

    if norm_W_proj < 1e-12:
        # If gradient projection is zero, any point in the ball is optimal for the linear subproblem.
        # Let's choose P_e = P_current to make the FW gap zero and stop.
        # Or, we can return Delta_e = 0, meaning P_e = P_nominal projected.
        Delta_e = np.zeros_like(P_nominal)
        # print("Warning: Gradient projection norm is near zero in direction finding.")
    else:
        # We want to minimize <W, Delta_e> = <W_proj, Delta_e>.
        # This is minimized when Delta_e is in the opposite direction of W_proj.
        Delta_e = -beta * W_proj / norm_W_proj

    # Construct P_e by adding delta to nominal and projecting
    P_e_target = P_nominal + Delta_e
    P_e = project_onto_l2_ball_and_stochastic(P_e_target, P_nominal, beta)

    # Calculate the Frank-Wolfe gap for *minimization*: <grad, P_current - P_e> = <W, P_current - P_e>
    # This gap should be >= 0 and converge to 0.
    fw_gap = np.sum(W * (P_current - P_e)) # Frobenius inner product

    # Ensure gap is non-negative (due to potential numerical issues)
    fw_gap = max(0, fw_gap)


    return P_e, fw_gap


def get_P_pi(policy, P_kernel, num_states, num_actions):
    """Compute P_π(s, s') = Σ_a π(a|s) P(s'|s,a)"""
    P_pi = np.zeros((num_states, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            if policy[s, a] > 0: # If action 'a' can be taken in state 's'
                P_pi[s, :] += policy[s, a] * P_kernel[s, a, :]
    return P_pi

def get_c_pi(policy, costs, num_states, num_actions):
    """Compute c_π(s) = Σ_a π(a|s) c(s,a)"""
    c_pi = np.zeros(num_states)
    for s in range(num_states):
        for a in range(num_actions):
            c_pi[s] += policy[s, a] * costs[s, a]
    return c_pi

def evaluate_policy_once(P_kernel, policy, costs, gamma, num_states, num_actions):
    """
    Evaluates a given policy π under a fixed transition kernel P.
    Returns V_pi, Q_pi, G_pi.
    Assumes costs are c(s,a)
    """
    P_pi = get_P_pi(policy, P_kernel, num_states, num_actions)
    c_pi_vec = get_c_pi(policy, costs, num_states, num_actions)

    # V_pi = (I - gamma * P_pi)^-1 * c_pi
    try:
        V_pi = np.linalg.solve(np.eye(num_states) - gamma * P_pi, c_pi_vec)
    except np.linalg.LinAlgError:
        print("Singular matrix in policy evaluation. Check P_pi.")
        # Fallback to pseudo-inverse, or handle error appropriately
        V_pi = np.linalg.pinv(np.eye(num_states) - gamma * P_pi) @ c_pi_vec

    Q_pi = np.zeros((num_states, num_actions))
    G_pi = np.zeros((num_states, num_actions, num_states)) # G_pi(s,a,s_next)

    for s in range(num_states):
        for a in range(num_actions):
            expected_future_value = 0
            for s_next in range(num_states):
                expected_future_value += P_kernel[s, a, s_next] * V_pi[s_next]
            Q_pi[s, a] = costs[s, a] + gamma * expected_future_value

            for s_next_g in range(num_states): # For G_pi
                G_pi[s, a, s_next_g] = costs[s, a] + gamma * V_pi[s_next_g] # From Lemma B.1 (iii)

    return V_pi, Q_pi, G_pi

def get_discounted_state_visitation(P_kernel, policy, gamma, s0, num_states, num_actions):
    """
    Computes d_π^P(s|s_0) = (1-γ) Σ_t γ^t P(S_t=s|S_0=s_0)
    d_pi^P( . | s0) = (1-gamma) * e_s0^T * (I - gamma * P_pi)^-1
    """
    P_pi = get_P_pi(policy, P_kernel, num_states, num_actions)
    
    # We need e_s0^T (I - gamma P_pi)^-1
    # (I - gamma P_pi)^T x = e_s0  => x = ((I - gamma P_pi)^T)^-1 e_s0
    # So d_s0 = (1-gamma) * x
    e_s0 = np.zeros(num_states)
    e_s0[s0] = 1.0
    
    try:
        inv_matrix = np.linalg.inv(np.eye(num_states) - gamma * P_pi.T) # Transpose for right multiply by e_s0 later
        d_s0 = (1 - gamma) * (inv_matrix @ e_s0)
    except np.linalg.LinAlgError:
        print("Singular matrix in visitation distribution. Check P_pi.")
        inv_matrix = np.linalg.pinv(np.eye(num_states) - gamma * P_pi.T)
        d_s0 = (1 - gamma) * (inv_matrix @ e_s0)
        
    return d_s0 # This is d_pi(s | s0) for all s

def get_adversary_advantage(G_pi, Q_pi):
    """ A_π^P(s,a,s') = G_π^P(s,a,s') - Q_π^P(s,a) """
    # G_pi is (num_states, num_actions, num_states_next)
    # Q_pi is (num_states, num_actions)
    # We need to broadcast Q_pi for the subtraction
    return G_pi - Q_pi[:, :, np.newaxis]

def cpi_policy_eval_p2(pi, P_nominal, R, gamma, mu, beta, S, A, tolerance=1e-7, max_iter=1000):
    """
    Implements Algorithm 3.2 for robust policy evaluation using the new CPI implementation.
    Adapted to work with the existing experiment framework interface.
    """
    set_policy(pi) # Make policy globally accessible for compatibility
    
    # Convert rewards to costs (negate for maximization)
    costs = -R  # Since we want to maximize rewards, we minimize negative rewards (costs)
    
    # Use uniform initial distribution if mu is uniform, otherwise pick the most likely state
    if np.allclose(mu, np.ones(S) / S):
        s0 = 0  # Use first state for uniform distribution
    else:
        s0 = np.argmax(mu)  # Use most likely initial state
    
    P_current = np.copy(P_nominal)
    iterations = []
    fw_gaps = []
    values = []
    best_value_so_far = np.inf  # Track minimum (worst-case) value
    
    start_time = time.time()
    
    for m in range(max_iter):
        # 1. Evaluate V_pi, Q_pi, G_pi under current P_current
        V_pi_m, Q_pi_m, G_pi_m = evaluate_policy_once(P_current, pi, costs, gamma, S, A)
        
        # Compute current value as V @ mu (convert back to rewards)
        current_value = (-V_pi_m) @ mu  # Negate back to rewards
        best_value_so_far = min(best_value_so_far, current_value)

        # 2. Compute discounted state visitation d_pi^P_m(s|s0)
        d_visitation_m = get_discounted_state_visitation(P_current, pi, gamma, s0, S, A)

        # 3. Compute adversary's advantage A_pi^P_m(s,a,s_next)
        Adv_m = get_adversary_advantage(G_pi_m, Q_pi_m)

        # 4. Find P_epsilon: The direction-finding subproblem
        P_epsilon = np.zeros_like(P_current)
        coefficients_for_P_new = np.zeros_like(P_current)

        for s_i in range(S):
            if d_visitation_m[s_i] == 0: # If state s_i is not reachable from s0
                P_epsilon[s_i, :, :] = P_current[s_i, :, :]
                continue
            for a_j in range(A):
                if pi[s_i, a_j] == 0: # If action a_j is not taken by policy at s_i
                    P_epsilon[s_i, a_j, :] = P_current[s_i, a_j, :]
                    continue

                # Coeff(s,a,s_next) = d_visitation_m[s] * pi[s,a] * Adv_m[s,a,s_next]
                current_coeffs = d_visitation_m[s_i] * pi[s_i, a_j] * Adv_m[s_i, a_j, :]
                coefficients_for_P_new[s_i, a_j, :] = current_coeffs

                # P_epsilon maximizes sum_s_next Coeff(s,a,s_next) * P_new(s_next|s,a)
                # by putting all mass on s_next_star
                s_next_star = np.argmax(current_coeffs)
                P_epsilon[s_i, a_j, s_next_star] = 1.0
        
        # 5. Compute Frank-Wolfe gap
        fw_gap_terms = coefficients_for_P_new * (P_epsilon - P_current)
        frank_wolfe_gap = (1.0 / (1.0 - gamma)) * np.sum(fw_gap_terms)

        iterations.append(m)
        fw_gaps.append(frank_wolfe_gap)
        values.append(current_value)

        if frank_wolfe_gap < tolerance:
            break

        # 6. Determine stepsize alpha_m
        if frank_wolfe_gap <= 0:
            alpha_m = 0.0
        else:
            if gamma == 0:
                alpha_m = 0.0
            else:
                alpha_m = frank_wolfe_gap * ((1 - gamma)**3) / (4 * (gamma**2))
        
        alpha_m = np.clip(alpha_m, 0.0, 1.0)

        # 7. Update P_current
        P_current = (1 - alpha_m) * P_current + alpha_m * P_epsilon
        
        # Normalize P_current to ensure valid probabilities
        for s_i in range(S):
            for a_j in range(A):
                row_sum = np.sum(P_current[s_i, a_j, :])
                if row_sum > 0:
                    P_current[s_i, a_j, :] /= row_sum
                else:
                    P_current[s_i, a_j, 0] = 1.0

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Final robust value is the best (minimum) value found during iterations
    robust_value = best_value_so_far

    # Handle final iteration data
    if m == max_iter - 1:
        if not iterations or iterations[-1] != m:
            V_final, _, _ = evaluate_policy_once(P_current, pi, costs, gamma, S, A)
            final_loop_value = (-V_final) @ mu  # Convert back to rewards
            robust_value = min(robust_value, final_loop_value)
            
            iterations.append(m)
            fw_gaps.append(frank_wolfe_gap)
            values.append(final_loop_value)

    convergence_data = {'iterations': iterations, 'gaps': fw_gaps, 'values': values}
    return robust_value, elapsed_time, convergence_data, len(iterations)

# #############################################
# ####### Experiment Setup           ##########
# #############################################

def generate_random_mdp(S, A, gamma=0.9):
    """Generates a random MDP."""
    # Random nominal transition kernel
    P_nominal = np.random.rand(S, A, S) + 0.1 # Add small value to avoid zero rows initially
    P_nominal = P_nominal / P_nominal.sum(axis=2, keepdims=True) # Normalize

    # Random reward function (e.g., in [0, 1])
    R = np.random.rand(S, A)

    # Uniform initial distribution
    mu = np.ones(S) / S

    # Uniform random policy
    pi = np.ones((S, A)) / A

    return P_nominal, R, mu, pi, gamma, S, A

def generate_garnet_mdp(S, A, gamma, b_param):
    """
    Generates a Garnet MDP.
    - S: Number of states
    - A: Number of actions
    - gamma: Discount factor
    - b_param: Branching parameter. For each (s,a), b_param next states are chosen.
               If b_param=1, transition is deterministic to one randomly chosen next state.
    """
    P_nominal = np.zeros((S, A, S))
    for s_idx in range(S):
        for a_idx in range(A):
            if b_param <= 0:
                # No outgoing transitions, or handle as error/specific case
                P_nominal[s_idx, a_idx, np.random.randint(S)] = 1.0 # Default to one random if b_param is invalid
            elif b_param == 1:
                # Deterministic transition to one randomly chosen state
                next_state = np.random.randint(S)
                P_nominal[s_idx, a_idx, next_state] = 1.0
            else:
                # Sample b_param unique next states
                num_successors = min(b_param, S)
                next_states_indices = np.random.choice(S, size=num_successors, replace=False)
                # Assign random probabilities to these states (summing to 1)
                probabilities = np.random.dirichlet(np.ones(num_successors))
                P_nominal[s_idx, a_idx, next_states_indices] = probabilities
    
    # Rewards c(s,a) from U[0,1]
    R = np.random.rand(S, A)

    # Uniform initial distribution
    mu = np.ones(S) / S

    # Uniform random policy for evaluation
    pi = np.ones((S, A)) / A

    return P_nominal, R, mu, pi, gamma, S, A

def sample_kernel_from_L2_ball(P_nominal, beta, S, A):
    """Samples a kernel approximately from the boundary of the L2 ball."""
    # 1. Sample random direction
    G = np.random.randn(S, A, S)

    # 2. Project rows to sum to zero to satisfy sum(Delta_sa) = 0 constraint
    row_sums_G = G.sum(axis=2, keepdims=True)
    Delta_unnorm = G - row_sums_G / S

    # 3. Normalize direction
    norm_Delta = np.linalg.norm(Delta_unnorm)
    if norm_Delta < 1e-12:
        return P_nominal # Avoid division by zero

    # 4. Scale to radius beta
    Delta = beta * Delta_unnorm / norm_Delta

    # 5. Add perturbation to nominal and project onto valid stochastic matrix within ball
    P_target = P_nominal + Delta
    P_sample = project_onto_l2_ball_and_stochastic(P_target, P_nominal, beta)

    return P_sample


# --- Start: Value Iteration Wrappers for Rectangular Models ---
def robust_value_iteration_sa_rect_L2(pi, P_nominal, R, mu, gamma, beta_sa_mat, tol, max_iter, S_val, A_val):
    """ Wrapper for iterative policy evaluation for SA-Rectangular L2 robust MDPs. """
    V_current_iter = np.zeros(S_val) # Renamed V_current to avoid conflict with outer scopes
    iter_history = []
    v_mu_history = []
    start_time = time.time()
    k_iter = 0
    for k_iter in range(max_iter):
        V_new_iter = policy_evaluation_sa_rectangular_L2(pi, R, P_nominal, gamma, beta_sa_mat, V_current_iter)
        iter_history.append(k_iter)
        v_mu_history.append(V_new_iter @ mu)
        if np.linalg.norm(V_new_iter - V_current_iter) < tol:
            break
        V_current_iter = V_new_iter
    
    elapsed_time = time.time() - start_time
    final_value = V_current_iter @ mu
    convergence_data = {'iterations': iter_history, 'values': v_mu_history}
    return final_value, elapsed_time, convergence_data, k_iter + 1

def robust_value_iteration_s_rect_L2(pi, P_nominal, R, mu, gamma, beta_s_vec, tol, max_iter, S_val, A_val):
    """ Wrapper for iterative policy evaluation for S-Rectangular L2 robust MDPs. """
    V_current_iter = np.zeros(S_val) # Renamed V_current
    iter_history = []
    v_mu_history = []
    start_time = time.time()
    k_iter = 0
    for k_iter in range(max_iter):
        V_new_iter = policy_evaluation_s_rectangular_L2(pi, R, P_nominal, gamma, beta_s_vec, V_current_iter)
        iter_history.append(k_iter)
        v_mu_history.append(V_new_iter @ mu)
        if np.linalg.norm(V_new_iter - V_current_iter) < tol:
            break
        V_current_iter = V_new_iter

    elapsed_time = time.time() - start_time
    final_value = V_current_iter @ mu
    convergence_data = {'iterations': iter_history, 'values': v_mu_history}
    return final_value, elapsed_time, convergence_data, k_iter + 1
# --- End: Value Iteration Wrappers ---

def kappa_2_variance(v_vector):
    """
    Calculates kappa_2(v) = ||v - mean(v)*1||_2.
    This is sqrt(sum_s (v(s) - mean(v))^2).

    Args:
        v_vector (np.array): The value function vector for S states.
    Returns:
        float: The value of kappa_2(v).
    """
    if not isinstance(v_vector, np.ndarray):
        v_vector = np.array(v_vector)
    
    S_len = len(v_vector) # Renamed S to S_len to avoid conflict
    if S_len == 0:
        return 0.0
    if S_len == 1: # No variance if only one state value
        return 0.0

    mean_v = np.mean(v_vector)
    return np.sqrt(np.sum((v_vector - mean_v)**2))

def policy_evaluation_sa_rectangular_L2(
    pi_policy, R0_nominal, P0_nominal, gamma,
    beta_sa, V_current_in # Changed V_current to V_current_in to avoid modifying input
):
    """
    Performs one step of policy evaluation for sa-rectangular L2 robust MDPs.
    (T_U^pi v)(s) = sum_a pi(a|s) [ - beta_s,a * kappa_2(v) + R0(s,a) + gamma * sum_s' P0(s'|s,a) v(s') ]
    """
    S_states, A_actions = pi_policy.shape # Renamed S,A to avoid conflict
    V_new = np.zeros(S_states)
    V_current = V_current_in # Use the input V_current

    current_kappa_2_V = kappa_2_variance(V_current)

    for s_idx in range(S_states): # Renamed s to s_idx
        expected_val_s = 0
        for a_idx in range(A_actions): # Renamed a to a_idx
            action_prob = pi_policy[s_idx, a_idx]
            if action_prob == 0:
                continue

            immediate_reward = R0_nominal[s_idx, a_idx]
            penalty_term = beta_sa[s_idx, a_idx] * current_kappa_2_V
            
            expected_future_value = 0
            for s_prime_idx in range(S_states): # Renamed s_prime to s_prime_idx
                expected_future_value += P0_nominal[s_idx, a_idx, s_prime_idx] * V_current[s_prime_idx]
            
            expected_val_s += action_prob * (
                immediate_reward - penalty_term + gamma * expected_future_value
            )
        V_new[s_idx] = expected_val_s
    return V_new

def policy_evaluation_s_rectangular_L2(
    pi_policy, R0_nominal, P0_nominal, gamma,
    beta_s, V_current_in # Changed V_current to V_current_in
):
    """
    Performs one step of policy evaluation for s-rectangular L2 robust MDPs.
    (T_U^pi v)(s) = - [gamma * beta_s * kappa_2(v)] * ||pi_s||_2 
                     + sum_a pi(a|s) [ R0(s,a) + gamma * sum_s' P0(s'|s,a) v(s') ]
    """
    S_states, A_actions = pi_policy.shape # Renamed S,A
    V_new = np.zeros(S_states)
    V_current = V_current_in # Use the input V_current

    current_kappa_2_V = kappa_2_variance(V_current)

    for s_idx in range(S_states): # Renamed s
        pi_s_vec = pi_policy[s_idx, :] # Renamed pi_s
        norm_pi_s_2 = np.sqrt(np.sum(pi_s_vec**2))
        
        penalty_factor_s = gamma * beta_s[s_idx] * current_kappa_2_V
        total_penalty_s = penalty_factor_s * norm_pi_s_2
        
        sum_pi_R_plus_gamma_PV = 0
        for a_idx in range(A_actions): # Renamed a
            action_prob = pi_policy[s_idx, a_idx]
            if action_prob == 0:
                continue
            
            immediate_reward = R0_nominal[s_idx, a_idx]
            expected_future_value = 0
            for s_prime_idx in range(S_states): # Renamed s_prime
                expected_future_value += P0_nominal[s_idx, a_idx, s_prime_idx] * V_current[s_prime_idx]
            
            sum_pi_R_plus_gamma_PV += action_prob * (
                immediate_reward + gamma * expected_future_value
            )
            


def run_single_mdp_comparison(S, A, beta, gamma=0.9, n_samples_opt=1000,
                             p1_tol=1e-6, p2_tol=1e-6, cpi_max_iter=100,
                             mdp_type="random", garnet_b_param=1, plot_convergence=True):
    """
    Compare all 4 algorithms on the SAME MDP instance.
    Shows direct algorithmic performance differences on identical problems.
    """
    print(f"\n🔍 Single MDP Comparison: S={S}, A={A}, β={beta}")
    print(f"   Using {n_samples_opt} samples for high accuracy benchmark")
    
    # Generate ONE MDP instance
    if mdp_type == "random":
        P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_random_mdp(S, A, gamma)
    elif mdp_type == "garnet":
        P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_garnet_mdp(S, A, gamma, garnet_b_param)
    else:
        raise ValueError(f"Unknown mdp_type: {mdp_type}")
    
    # Generate a random policy for evaluation
    np.random.seed(42)  # Fixed seed for reproducibility
    pi = np.random.rand(S, A)
    pi = pi / pi.sum(axis=1, keepdims=True)  # Normalize to stochastic
    
    print(f"   Generated {mdp_type} MDP and random policy")
    
    # Compute high-accuracy benchmark via sampling
    print(f"   Computing benchmark via {n_samples_opt} samples...")
    benchmark_values = []
    for _ in range(n_samples_opt):
        P_sample = sample_kernel_from_L2_ball(P_nominal, beta, S, A)
        P_pi_sample = build_P_pi(P_sample, pi, S, A)
        R_pi_sample = np.sum(pi * R, axis=1)
        try:
            V_sample = compute_V(P_pi_sample, R_pi_sample, gamma, S)
            benchmark_values.append(mu @ V_sample)
        except:
            continue
    
    if not benchmark_values:
        print("   ❌ Failed to compute benchmark - no valid samples")
        return
    
    benchmark_min = min(benchmark_values)
    benchmark_mean = np.mean(benchmark_values)
    print(f"   ✅ Benchmark: min={benchmark_min:.6f}, mean={benchmark_mean:.6f}")
    
    # Run all 4 algorithms on the same MDP
    algorithms = {
        "Ours (Binary+Spectral)": lambda: binary_search_policy_eval_p1(
            pi, P_nominal, R, gamma_trial, mu, beta, S_trial, A_trial, tolerance=p1_tol, max_iter=100),
        "CPI (Frank-Wolfe)": lambda: cpi_policy_eval_p2(
            pi, P_nominal, R, gamma_trial, mu, beta, S_trial, A_trial, tolerance=p2_tol, max_iter=cpi_max_iter),
        "SA-Rectangular L2": lambda: robust_value_iteration_sa_rect_L2(
            pi, P_nominal, R, mu, gamma_trial, beta * np.ones((S_trial, A_trial)), p2_tol, cpi_max_iter, S_trial, A_trial),
        "S-Rectangular L2": lambda: robust_value_iteration_s_rect_L2(
            pi, P_nominal, R, mu, gamma_trial, beta * np.ones(S_trial), p2_tol, cpi_max_iter, S_trial, A_trial)
    }
    
    results = {}
    convergence_data = {}
    
    for name, algo_func in algorithms.items():
        print(f"\n   Running {name}...")
        start_time = time.time()
        
        try:
            result = algo_func()
            end_time = time.time()
            
            # Handle different return formats
            if name in ["SA-Rectangular L2", "S-Rectangular L2"]:
                # These return (value, time, conv_data, iterations)
                value, algo_time, conv_info, iterations = result
            elif isinstance(result, tuple) and len(result) >= 2:
                # Binary search and CPI return (value, time, conv_data, iterations)
                if len(result) == 4:
                    value, algo_time, conv_info, iterations = result
                else:
                    value, conv_info = result[0], result[1] if len(result) > 1 else None
                    iterations = conv_info.get('iterations', 'N/A') if conv_info and isinstance(conv_info, dict) else 'N/A'
            else:
                # Single value return
                value, conv_info, iterations = result, None, 'N/A'
            
            gap = value - benchmark_min
            
            results[name] = {
                'value': value,
                'time': end_time - start_time,
                'gap': gap,
                'iterations': iterations,
                'converged': True  # Assume converged if we got a result
            }
            
            if conv_info and isinstance(conv_info, dict) and 'values' in conv_info:
                convergence_data[name] = conv_info
            
            print(f"     ✅ Value: {value:.6f}, Time: {end_time - start_time:.3f}s, Gap: {gap:.6f}")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results[name] = {
                'value': None, 'time': None, 'gap': None, 
                'iterations': 'Failed', 'converged': False
            }
    
    # Print comparison table
    print(f"\n📊 RESULTS COMPARISON (S={S_trial}, A={A_trial}, β={beta})")
    print("="*80)
    print(f"{'Algorithm':<25} {'Value':<12} {'Time(s)':<10} {'Gap':<12} {'Iters':<8} {'Status'}")
    print("-"*80)
    print(f"{'Benchmark (min)':<25} {benchmark_min:<12.6f} {'-':<10} {0.0:<12.6f} {'-':<8} {'Reference'}")
    
    for name, res in results.items():
        if res['value'] is not None:
            status = "✅ Conv" if res['converged'] else "⚠️ Max"
            print(f"{name:<25} {res['value']:<12.6f} {res['time']:<10.3f} {res['gap']:<12.6f} {res['iterations']:<8} {status}")
        else:
            print(f"{name:<25} {'Failed':<12} {'Failed':<10} {'Failed':<12} {'Failed':<8} {'❌ Error'}")
    
    # Plot convergence if requested and data available
    if plot_convergence and convergence_data:
        plt.figure(figsize=(12, 8))
        
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        linestyles = ['-', '--', ':', '-.']
        
        # Find the maximum number of iterations across all algorithms
        max_iterations = 0
        for name, conv_info in convergence_data.items():
            if 'values' in conv_info and 'iterations' in conv_info and conv_info['iterations']:
                max_iterations = max(max_iterations, max(conv_info['iterations']) + 1)
        
        # If no iterations found, set a default
        if max_iterations == 0:
            max_iterations = 100
        
        for i, (name, conv_info) in enumerate(convergence_data.items()):
            if 'values' in conv_info and 'iterations' in conv_info and conv_info['values']:
                iterations = conv_info['iterations']
                values = conv_info['values']
                
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                
                # Plot the actual convergence trajectory
                plt.plot(iterations, values, 
                        color=color, linestyle=linestyle,
                        marker='o', markersize=4, linewidth=2,
                        label=name)
                
                # If algorithm converged before max_iterations, extend the line
                if iterations and len(iterations) > 0:
                    last_iteration = max(iterations)
                    last_value = values[-1]  # Final converged value
                    
                    if last_iteration < max_iterations - 1:
                        # Create extended line with the converged value
                        extended_iterations = list(range(last_iteration + 1, max_iterations))
                        extended_values = [last_value] * len(extended_iterations)
                        
                        # Plot extended line with dotted style AND markers to continue the dots
                        plt.plot(extended_iterations, extended_values, 
                                color=color, linestyle=':', linewidth=1.5, alpha=0.7,
                                marker='o', markersize=3)  # Smaller markers for post-convergence
                        
                        # Mark the convergence point with a special marker
                        plt.plot(last_iteration, last_value, 
                                marker='*', markersize=12, color=color, 
                                markeredgecolor='black', markeredgewidth=1,
                                label=f'{name} converged')
        
        plt.axhline(benchmark_min, color='red', linestyle=':', linewidth=2, 
                   label=f'Benchmark Min = {benchmark_min:.6f}')
        # Removed benchmark mean line - keeping only the min
        
        plt.xlabel('Iteration')
        plt.ylabel('Robust Value Estimate')
        plt.title(f'MDP Convergence Comparison\n(S={S_trial}, A={A_trial}, β={beta}, {mdp_type} MDP)')
        
        # Create a custom legend to avoid duplicate entries
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out duplicate convergence markers from legend
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if 'converged' not in label:
                unique_labels[label] = handle
        
        # Add convergence info to legend
        legend_entries = list(unique_labels.items())
        legend_entries.append((plt.Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=10), 
                              'Convergence Point'))
        legend_entries.append((plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5), 
                              'Post-Convergence'))
        
        plt.legend([h for _, h in legend_entries], [l for l, _ in legend_entries], 
                  loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return results


def run_performance_analysis(S, A, beta, gamma=0.9, n_trials=20, n_samples_opt=100,
                           p1_tol=1e-6, p2_tol=1e-6, cpi_max_iter=100,
                           mdp_type="random", garnet_b_param=1):
    """
    Run multiple trials across different random MDPs to get statistical measures
    of convergence speed, accuracy, and reliability.
    """
    print(f"\n📈 Performance Analysis: S={S}, A={A}, β={beta}")
    print(f"   Running {n_trials} trials with {n_samples_opt} samples each")
    
    algorithms = {
        "Ours": lambda pi, P, R, mu: binary_search_policy_eval_p1(
            pi, P, R, gamma, mu, beta, S, A, tolerance=p1_tol, max_iter=100),
        "CPI": lambda pi, P, R, mu: cpi_policy_eval_p2(
            pi, P, R, gamma, mu, beta, S, A, tolerance=p2_tol, max_iter=cpi_max_iter),
        "SA-Rect": lambda pi, P, R, mu: robust_value_iteration_sa_rect_L2(
            pi, P, R, mu, gamma, beta * np.ones((S, A)), p2_tol, cpi_max_iter, S, A),
        "S-Rect": lambda pi, P, R, mu: robust_value_iteration_s_rect_L2(
            pi, P, R, mu, gamma, beta * np.ones(S), p2_tol, cpi_max_iter, S, A)
    }
    
    # Collect results across trials
    trial_results = {name: {'times': [], 'gaps': [], 'iterations': [], 'successes': 0} 
                    for name in algorithms.keys()}
    
    for trial in range(n_trials):
        if trial % 5 == 0:
            print(f"   Trial {trial+1}/{n_trials}...")
        
        # Generate new MDP for this trial
        if mdp_type == "random":
            P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_random_mdp(S, A, gamma)
        elif mdp_type == "garnet":
            P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_garnet_mdp(S, A, gamma, garnet_b_param)
        else:
            raise ValueError(f"Unknown mdp_type: {mdp_type}")
        
        # Generate random policy
        np.random.seed(trial + 1000)  # Different seed per trial
        pi_custom = np.random.rand(S, A)  # Use S, A from function parameters  
        pi_custom = pi_custom / pi_custom.sum(axis=1, keepdims=True)
        
        # Compute benchmark for this trial (fewer samples for speed)
        benchmark_values = []
        for _ in range(n_samples_opt):
            try:
                P_sample = sample_kernel_from_L2_ball(P_nominal, beta, S_trial, A_trial)
                P_pi_sample = build_P_pi(P_sample, pi_custom, S_trial, A_trial)
                R_pi_sample = np.sum(pi_custom * R, axis=1)
                V_sample = compute_V(P_pi_sample, R_pi_sample, gamma_trial, S_trial)
                benchmark_values.append(mu @ V_sample)
            except:
                continue
        
        if not benchmark_values:
            continue  # Skip this trial if benchmark failed
        
        benchmark_min = min(benchmark_values)
        
        # Test each algorithm on this trial
        for name, algo_func in algorithms.items():
            try:
                start_time = time.time()
                result = algo_func(pi_custom, P_nominal, R, mu)
                end_time = time.time()
                
                # Handle different return formats
                if name in ["SA-Rect", "S-Rect"]:
                    # These return (value, time, conv_data, iterations)
                    value, algo_time, conv_info, iterations = result
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Binary search and CPI return (value, time, conv_data, iterations)
                    if len(result) == 4:
                        value, algo_time, conv_info, iterations = result
                    else:
                        value, conv_info = result[0], result[1] if len(result) > 1 else None
                        iterations = conv_info.get('iterations', 0) if conv_info and isinstance(conv_info, dict) else 0
                else:
                    # Single value return
                    value, conv_info, iterations = result, None, 0
                
                # Record metrics
                trial_results[name]['times'].append(end_time - start_time)
                trial_results[name]['gaps'].append(value - benchmark_min)
                trial_results[name]['iterations'].append(iterations)
                trial_results[name]['successes'] += 1
                
            except Exception as e:
                # Algorithm failed on this trial
                pass
    
    # Compute and display statistics
    print(f"\n📊 PERFORMANCE ANALYSIS RESULTS (S={S}, A={A}, β={beta})")
    print("="*90)
    print(f"{'Algorithm':<12} {'Success%':<9} {'Avg Time':<10} {'Avg Gap':<12} {'Avg Iters':<11} {'Time Std':<10}")
    print("-"*90)
    
    stats_summary = {}
    for name, data in trial_results.items():
        if data['successes'] > 0:
            success_rate = (data['successes'] / n_trials) * 100
            avg_time = np.mean(data['times'])
            avg_gap = np.mean(data['gaps'])
            avg_iters = np.mean(data['iterations'])
            std_time = np.std(data['times'])
            
            stats_summary[name] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_gap': avg_gap,
                'avg_iterations': avg_iters,
                'std_time': std_time
            }
            
            print(f"{name:<12} {success_rate:<9.1f} {avg_time:<10.3f} {avg_gap:<12.6f} {avg_iters:<11.1f} {std_time:<10.3f}")
        else:
            print(f"{name:<12} {'0.0':<9} {'Failed':<10} {'Failed':<12} {'Failed':<11} {'Failed':<10}")
    
    # Create box plots for performance comparison
    if any(data['successes'] > 0 for data in trial_results.values()):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time comparison
        time_data = [data['times'] for name, data in trial_results.items() if data['times']]
        time_labels = [name for name, data in trial_results.items() if data['times']]
        if time_data:
            axes[0,0].boxplot(time_data, labels=time_labels)
            axes[0,0].set_ylabel('Time (seconds)')
            axes[0,0].set_title('Computation Time Distribution')
            axes[0,0].grid(True, alpha=0.3)
        
        # Gap comparison
        gap_data = [data['gaps'] for name, data in trial_results.items() if data['gaps']]
        gap_labels = [name for name, data in trial_results.items() if data['gaps']]
        if gap_data:
            axes[0,1].boxplot(gap_data, labels=gap_labels)
            axes[0,1].set_ylabel('Gap from Benchmark')
            axes[0,1].set_title('Accuracy Distribution (Lower is Better)')
            axes[0,1].grid(True, alpha=0.3)
        
        # Iterations comparison
        iter_data = [data['iterations'] for name, data in trial_results.items() if data['iterations']]
        iter_labels = [name for name, data in trial_results.items() if data['iterations']]
        if iter_data:
            axes[1,0].boxplot(iter_data, labels=iter_labels)
            axes[1,0].set_ylabel('Iterations to Convergence')
            axes[1,0].set_title('Convergence Speed Distribution')
            axes[1,0].grid(True, alpha=0.3)
        
        # Success rate bar chart
        success_rates = [stats_summary.get(name, {}).get('success_rate', 0) 
                        for name in algorithms.keys()]
        axes[1,1].bar(algorithms.keys(), success_rates, alpha=0.7)
        axes[1,1].set_ylabel('Success Rate (%)')
        axes[1,1].set_title('Algorithm Reliability')
        axes[1,1].set_ylim(0, 100)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Analysis: S={S}, A={A}, β={beta}\n({n_trials} trials, {mdp_type} MDPs)')
        plt.tight_layout()
        plt.show()
    
    return stats_summary


def run_experiments(S, A, beta, gamma=0.9, n_trials=10, n_samples_opt=1000,
                    p1_tol=1e-6, p2_tol=1e-6, cpi_max_iter=1000,
                    mdp_type="random", garnet_b_param=1, 
                    plot_single_trial_convergence=False,
                    plot_aggregated_convergence=True,
                    plot_time_comparison=False,
                    plot_gap_comparison=False):
    """Runs the comparison experiments."""

    print(f"\n{'='*10} Running Experiments ({mdp_type.capitalize()}) {'='*10}")
    print(f"Params: S={S}, A={A}, beta={beta}, gamma={gamma}, n_trials={n_trials}")
    if mdp_type == "garnet":
        print(f"Garnet b_param={garnet_b_param}")
    print(f"n_samples_opt={n_samples_opt}, p1_tol={p1_tol}, p2_tol={p2_tol}, cpi_max_iter={cpi_max_iter}")
    print(f"{'='* (39 if mdp_type=='random' else 48)}")


    times_p1 = []
    times_p2 = []
    vals_p1 = []
    vals_p2 = []
    gaps_p1 = []
    gaps_p2 = []
    iters_p1 = []
    iters_p2 = []

    times_sa_rect = []
    vals_sa_rect = []
    gaps_sa_rect = []
    iters_sa_rect = []
    all_conv_data_sa_rect = []

    times_s_rect = []
    vals_s_rect = []
    gaps_s_rect = []
    iters_s_rect = []
    all_conv_data_s_rect = []

    all_conv_data_p1 = []
    all_conv_data_p2 = []
    all_v_opt_approx = []

    for i in range(n_trials):
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        if mdp_type == "garnet":
            P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_garnet_mdp(S, A, gamma, garnet_b_param)
            print(f"Generated Garnet MDP (b={garnet_b_param}) for S={S_trial}, A={A_trial}")
        else: # Default to random
            P_nominal, R, mu, pi, gamma_trial, S_trial, A_trial = generate_random_mdp(S, A, gamma)
            print(f"Generated Random MDP for S={S_trial}, A={A_trial}")

        # --- Approximate (Worst-Case) Value (Benchmark) ---
        print(f"Benchmarking: Sampling {n_samples_opt} kernels to find worst-case...")
        sampled_values = []
        # Use tqdm for progress bar if available
        for _ in tqdm(range(n_samples_opt), desc="Benchmark Sampling"):
            P_sample = sample_kernel_from_L2_ball(P_nominal, beta, S_trial, A_trial)
            P_pi_sample = build_P_pi(P_sample, pi, S_trial, A_trial)
            R_pi = np.sum(pi * R, axis=1)
            try:
                V_sample = compute_V(P_pi_sample, R_pi, gamma_trial, S_trial)
                sampled_values.append(V_sample @ mu)
            except Exception as e:
                # print(f"Warning: Value computation failed for a sample: {e}")
                sampled_values.append(np.inf) # Penalize failed samples for minimization

        # Filter out infinities before taking the minimum
        valid_sampled_values = [v for v in sampled_values if not np.isinf(v)]

        if not valid_sampled_values:
             print("Warning: Benchmark sampling failed to produce valid values. Using nominal value as approx.")
             P_pi_nom = build_P_pi(P_nominal, pi, S_trial, A_trial)
             R_pi_nom = np.sum(pi*R, axis=1)
             try:
                 V_nom = compute_V(P_pi_nom, R_pi_nom, gamma_trial, S_trial)
                 V_worst_approx = V_nom @ mu
             except Exception as e:
                 print(f"Error computing nominal value as fallback: {e}. Setting benchmark to NaN.")
                 V_worst_approx = np.nan
        else:
            # Find the minimum value among samples as the robust value estimate
            V_worst_approx = np.min(valid_sampled_values)

        all_v_opt_approx.append(V_worst_approx)
        if not np.isnan(V_worst_approx):
            print(f"Approximate Robust Value (Benchmark Min): {V_worst_approx:.7f}")
        else:
             print("Approximate Robust Value (Benchmark Min): Failed")


        # --- Run Paper 1 Algorithm ---
        print("Running Paper 1 (Binary Search + Spectral)...")
        val_p1, time_p1, conv_data_p1, iter_p1 = binary_search_policy_eval_p1(
            pi, P_nominal, R, gamma_trial, mu, beta, S_trial, A_trial, tolerance=p1_tol
        )
        if not np.isnan(val_p1) and not np.isnan(V_worst_approx):
            times_p1.append(time_p1)
            vals_p1.append(val_p1)
            # Gap = Algo Value - Benchmark Min Value (should be >= 0)
            gap_p1 = val_p1 - V_worst_approx
            gaps_p1.append(gap_p1)
            iters_p1.append(iter_p1)
            all_conv_data_p1.append(conv_data_p1)
            print(f"  Result: Value={val_p1:.7f}, Time={time_p1:.4f}s, Gap={gap_p1:.7f}, Iters={iter_p1}")
        elif np.isnan(val_p1):
            print("  Result: Failed to compute.")
            all_conv_data_p1.append(None) # Placeholder for failed trial
        else: # V_worst_approx is nan
             times_p1.append(time_p1)
             vals_p1.append(val_p1)
             gaps_p1.append(np.nan) # Gap is undefined
             iters_p1.append(iter_p1)
             all_conv_data_p1.append(conv_data_p1)
             print(f"  Result: Value={val_p1:.7f}, Time={time_p1:.4f}s, Gap=NaN, Iters={iter_p1}")


        # --- Run Paper 2 Algorithm ---
        print("Running Paper 2 (CPI / Frank-Wolfe)...")
        val_p2, time_p2, conv_data_p2, iter_p2 = cpi_policy_eval_p2(
            pi, P_nominal, R, gamma_trial, mu, beta, S_trial, A_trial, tolerance=p2_tol, max_iter=cpi_max_iter
        )
        if not np.isnan(val_p2) and not np.isnan(V_worst_approx):
            times_p2.append(time_p2)
            vals_p2.append(val_p2)
            # Gap = Algo Value - Benchmark Min Value (should be >= 0)
            gap_p2 = val_p2 - V_worst_approx
            gaps_p2.append(gap_p2)
            iters_p2.append(iter_p2)
            all_conv_data_p2.append(conv_data_p2)
            print(f"  Result: Value={val_p2:.7f}, Time={time_p2:.4f}s, Gap={gap_p2:.7f}, Iters={iter_p2}")
        elif np.isnan(val_p2):
             print("  Result: Failed to compute.")
             all_conv_data_p2.append(None) # Placeholder for failed trial
        else: # V_worst_approx is nan
             times_p2.append(time_p2)
             vals_p2.append(val_p2)
             gaps_p2.append(np.nan) # Gap is undefined
             iters_p2.append(iter_p2)
             all_conv_data_p2.append(conv_data_p2)
             print(f"  Result: Value={val_p2:.7f}, Time={time_p2:.4f}s, Gap=NaN, Iters={iter_p2}")


        # --- Run SA-Rectangular L2 Value Iteration ---
        print("Running SA-Rectangular L2 (Value Iteration)...")
        # For SA-Rectangular, beta_sa is a (S, A) matrix
        beta_sa_mat_trial = np.full((S_trial, A_trial), 0.01)
        val_sa, time_sa, conv_data_sa, iter_sa = robust_value_iteration_sa_rect_L2(
            pi, P_nominal, R, mu, gamma_trial, beta_sa_mat_trial, p2_tol, cpi_max_iter, S_trial, A_trial
        )
        if not np.isnan(val_sa) and not np.isnan(V_worst_approx):
            times_sa_rect.append(time_sa)
            vals_sa_rect.append(val_sa)
            gap_sa = val_sa - V_worst_approx
            gaps_sa_rect.append(gap_sa)
            iters_sa_rect.append(iter_sa)
            all_conv_data_sa_rect.append(conv_data_sa)
            print(f"  Result: Value={val_sa:.7f}, Time={time_sa:.4f}s, Gap={gap_sa:.7f}, Iters={iter_sa}")
        elif np.isnan(val_sa):
            print("  Result: Failed to compute.")
            all_conv_data_sa_rect.append(None)
        else: # V_worst_approx is nan
            times_sa_rect.append(time_sa)
            vals_sa_rect.append(val_sa)
            gaps_sa_rect.append(np.nan)
            iters_sa_rect.append(iter_sa)
            all_conv_data_sa_rect.append(conv_data_sa)
            print(f"  Result: Value={val_sa:.7f}, Time={time_sa:.4f}s, Gap=NaN, Iters={iter_sa}")

        # --- Run S-Rectangular L2 Value Iteration ---
        print("Running S-Rectangular L2 (Value Iteration)...")
        # For S-Rectangular, beta_s is a (S,) vector
        beta_s_vec_trial = np.full(S_trial, 0.01)
        val_s, time_s, conv_data_s, iter_s = robust_value_iteration_s_rect_L2(
            pi, P_nominal, R, mu, gamma_trial, beta_s_vec_trial, p2_tol, cpi_max_iter, S_trial, A_trial
        )
        if not np.isnan(val_s) and not np.isnan(V_worst_approx):
            times_s_rect.append(time_s)
            vals_s_rect.append(val_s)
            gap_s = val_s - V_worst_approx
            gaps_s_rect.append(gap_s)
            iters_s_rect.append(iter_s)
            all_conv_data_s_rect.append(conv_data_s)
            print(f"  Result: Value={val_s:.7f}, Time={time_s:.4f}s, Gap={gap_s:.7f}, Iters={iter_s}")
        elif np.isnan(val_s):
            print("  Result: Failed to compute.")
            all_conv_data_s_rect.append(None)
        else: # V_worst_approx is nan
            times_s_rect.append(time_s)
            vals_s_rect.append(val_s)
            gaps_s_rect.append(np.nan)
            iters_s_rect.append(iter_s)
            all_conv_data_s_rect.append(conv_data_s)
            print(f"  Result: Value={val_s:.7f}, Time={time_s:.4f}s, Gap=NaN, Iters={iter_s}")


    # --- Analysis and Plotting ---
    print(f"\n{'='*10} Results Summary (Avg over {len(times_p1) + len(times_p2) + len(times_sa_rect) + len(times_s_rect)} successful runs) {'='*10}")
    valid_benchmarks = [v for v in all_v_opt_approx if not np.isnan(v)]
    if valid_benchmarks:
        avg_benchmark = np.mean(valid_benchmarks)
        print(f"Avg Benchmark Value (Min): {avg_benchmark:.7f}")
    else:
        avg_benchmark = np.nan
        print("Avg Benchmark Value (Min): N/A (No valid benchmarks)")


    # --- Prepare Results for Saving --- 
    results = {
        'params': {
            'S': S, 'A': A, 'beta': beta, 'gamma': gamma, 'n_trials': n_trials,
            'n_samples_opt': n_samples_opt, 'p1_tol': p1_tol, 'p2_tol': p2_tol,
            'cpi_max_iter': cpi_max_iter, 'mdp_type': mdp_type, 'garnet_b_param': garnet_b_param
        },
        'times_p1': times_p1,
        'times_p2': times_p2,
        'vals_p1': vals_p1,
        'vals_p2': vals_p2,
        'gaps_p1': gaps_p1,
        'gaps_p2': gaps_p2,
        'iters_p1': iters_p1,
        'iters_p2': iters_p2,
        'all_v_opt_approx': all_v_opt_approx,
        'all_conv_data_p1': all_conv_data_p1,
        'all_conv_data_p2': all_conv_data_p2,
        'avg_benchmark': avg_benchmark,

        'times_sa_rect': times_sa_rect,
        'vals_sa_rect': vals_sa_rect,
        'gaps_sa_rect': gaps_sa_rect,
        'iters_sa_rect': iters_sa_rect,
        'all_conv_data_sa_rect': all_conv_data_sa_rect,

        'times_s_rect': times_s_rect,
        'vals_s_rect': vals_s_rect,
        'gaps_s_rect': gaps_s_rect,
        'iters_s_rect': iters_s_rect,
        'all_conv_data_s_rect': all_conv_data_s_rect,
    }

    # Helper for JSON serialization (handles numpy arrays, NaN, Inf)
    def json_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                           np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(obj):
                return 'NaN' # Use string for NaN
            elif np.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
            return float(obj)
        elif isinstance(obj, (np.bool_)):
             return bool(obj)
        elif isinstance(obj, (np.void)):
             return None # Or other representation if needed
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    # Generate Filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"experiment_results_{mdp_type}"
    if mdp_type == "garnet":
        filename_prefix += f"_b{garnet_b_param}"
    filename = f"{filename_prefix}_S{S}_A{A}_beta{beta}_{timestamp}.json"

    # Save to JSON
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, default=json_converter)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"\nError saving results to {filename}: {e}")

    if times_p1:
        print(f"\nOurs (Binary Search):")
        print(f"  Avg Time: {np.mean(times_p1):.4f} +/- {np.std(times_p1):.4f} s")
        print(f"  Avg Value: {np.mean(vals_p1):.7f} +/- {np.std(vals_p1):.7f}")
        valid_gaps_p1 = [g for g in gaps_p1 if not np.isnan(g)]
        if valid_gaps_p1:
             print(f"  Avg Gap: {np.mean(valid_gaps_p1):.7f} +/- {np.std(valid_gaps_p1):.7f}")
        else:
             print(f"  Avg Gap: N/A")

        print(f"  Avg Iters: {np.mean(iters_p1):.1f}")
    else:
        print("\nOurs (Binary Search): No successful runs.")

    if times_p2:
        print(f"\nCPI (Frank-Wolfe):")
        print(f"  Avg Time: {np.mean(times_p2):.4f} +/- {np.std(times_p2):.4f} s")
        print(f"  Avg Value: {np.mean(vals_p2):.7f} +/- {np.std(vals_p2):.7f}")
        valid_gaps_p2 = [g for g in gaps_p2 if not np.isnan(g)]
        if valid_gaps_p2:
             print(f"  Avg Gap: {np.mean(valid_gaps_p2):.7f} +/- {np.std(valid_gaps_p2):.7f}")
        else:
             print(f"  Avg Gap: N/A")
        print(f"  Avg Iters: {np.mean(iters_p2):.1f}")
    else:
         print("\nCPI (Frank-Wolfe): No successful runs.")

    if times_sa_rect:
        print(f"\nSA-Rectangular L2 (VI):")
        print(f"  Avg Time: {np.mean(times_sa_rect):.4f} +/- {np.std(times_sa_rect):.4f} s")
        print(f"  Avg Value: {np.mean(vals_sa_rect):.7f} +/- {np.std(vals_sa_rect):.7f}")
        valid_gaps_sa = [g for g in gaps_sa_rect if not np.isnan(g)]
        if valid_gaps_sa:
             print(f"  Avg Gap: {np.mean(valid_gaps_sa):.7f} +/- {np.std(valid_gaps_sa):.7f}")
        else:
             print(f"  Avg Gap: N/A")
        print(f"  Avg Iters: {np.mean(iters_sa_rect):.1f}")
    else:
        print("\nSA-Rectangular L2 (VI): No successful runs.")

    if times_s_rect:
        print(f"\nS-Rectangular L2 (VI):")
        print(f"  Avg Time: {np.mean(times_s_rect):.4f} +/- {np.std(times_s_rect):.4f} s")
        print(f"  Avg Value: {np.mean(vals_s_rect):.7f} +/- {np.std(vals_s_rect):.7f}")
        valid_gaps_s = [g for g in gaps_s_rect if not np.isnan(g)]
        if valid_gaps_s:
             print(f"  Avg Gap: {np.mean(valid_gaps_s):.7f} +/- {np.std(valid_gaps_s):.7f}")
        else:
             print(f"  Avg Gap: N/A")
        print(f"  Avg Iters: {np.mean(iters_s_rect):.1f}")
    else:
        print("\nS-Rectangular L2 (VI): No successful runs.")

    print(f"{'='*53}")


    # Plotting
    # Find the first successful trial for convergence plots
    first_successful_trial_idx = -1
    for idx, (c1, c2, c_sa, c_s, bench) in enumerate(zip(all_conv_data_p1, all_conv_data_p2, all_conv_data_sa_rect, all_conv_data_s_rect, all_v_opt_approx)):
        if c1 is not None and c2 is not None and c_sa is not None and c_s is not None and not np.isnan(bench):
            first_successful_trial_idx = idx
            break

    if plot_single_trial_convergence and first_successful_trial_idx != -1:
        print(f"\nPlotting convergence for Trial {first_successful_trial_idx + 1}...")
        plt.figure(figsize=(10, 6))
        conv_p1 = all_conv_data_p1[first_successful_trial_idx]
        conv_p2 = all_conv_data_p2[first_successful_trial_idx]
        bench_val = all_v_opt_approx[first_successful_trial_idx]
        conv_sa = all_conv_data_sa_rect[first_successful_trial_idx]
        conv_s = all_conv_data_s_rect[first_successful_trial_idx]

        # Plot Paper 1 Convergence
        if conv_p1 and conv_p1['iterations']:
            plt.plot(conv_p1['iterations'], conv_p1['values'], marker='.', linestyle='-', markersize=4, label='Ours (Binary Search)')

        # Plot Paper 2 Convergence
        if conv_p2 and conv_p2['iterations']:
             # Ensure CPI values are plotted correctly (value at iter k is computed *before* update k)
             plt.plot(conv_p2['iterations'], conv_p2['values'], marker='x', linestyle='--', markersize=4, label='CPI (Frank-Wolfe)')

        # Plot SA-Rectangular Convergence
        if conv_sa and conv_sa['iterations']:
            plt.plot(conv_sa['iterations'], conv_sa['values'], marker='s', linestyle=':', markersize=4, label='SA-Rect L2 (VI)')
        
        # Plot S-Rectangular Convergence
        if conv_s and conv_s['iterations']:
            plt.plot(conv_s['iterations'], conv_s['values'], marker='^', linestyle='-.', markersize=4, label='S-Rect L2 (VI)')


        # Plot Benchmark Line
        plt.axhline(bench_val, color='r', linestyle=':', linewidth=2, label=f'Benchmark Min ({bench_val:.4f})')

        plt.xlabel("Iteration")
        plt.ylabel("Estimated Robust Value")
        plt.title(f'Single MDP Convergence Comparison\n(S={S_trial}, A={A_trial}, β={beta}, {mdp_type} MDP)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    elif plot_single_trial_convergence:
        print("No successful trial found with data for all algorithms to plot single-trial convergence.")


    # Plotting Box Plots only if there's data for all algorithms
    plot_labels_all = []
    data_times_all = []
    data_gaps_all = []

    if times_p1:
        plot_labels_all.append('Ours (Non-Rect)')
        data_times_all.append(times_p1)
        data_gaps_all.append([g for g in gaps_p1 if not np.isnan(g)])
    if times_p2:
        plot_labels_all.append('CPI (Non-Rect)')
        data_times_all.append(times_p2)
        data_gaps_all.append([g for g in gaps_p2 if not np.isnan(g)])
    if times_sa_rect:
        plot_labels_all.append('SA-Rect L2')
        data_times_all.append(times_sa_rect)
        data_gaps_all.append([g for g in gaps_sa_rect if not np.isnan(g)])
    if times_s_rect:
        plot_labels_all.append('S-Rect L2')
        data_times_all.append(times_s_rect)
        data_gaps_all.append([g for g in gaps_s_rect if not np.isnan(g)])


    if len(data_times_all) >= 1 and plot_aggregated_convergence: # Plot if at least one algorithm has data
        # ====================== NEW: ERROR-BAND CONVERGENCE ======================
        # Aggregate per-iteration statistics across trials (mean ± std)
        def aggregate_conv(conv_list):
            # conv_list : list of dicts (maybe None) with keys 'iterations', 'values'
            filtered = [c for c in conv_list if c is not None and c['iterations']]
            if not filtered:
                return None, None, None
            # Determine the maximum iteration index observed among all trials
            max_len = max(len(c['values']) for c in filtered)
            # Collect per-iteration values padding with last value (to retain plateau)
            arr = np.full((len(filtered), max_len), np.nan)
            for r, c in enumerate(filtered):
                vals = c['values']
                # pad with last value for remaining timesteps
                arr[r, : len(vals)] = vals
                if len(vals) < max_len:
                    arr[r, len(vals) :] = vals[-1]
            mean_vals = np.nanmean(arr, axis=0)
            std_vals = np.nanstd(arr, axis=0)
            return np.arange(max_len), mean_vals, std_vals

        it_p1, mean_p1, std_p1 = aggregate_conv(all_conv_data_p1)
        it_p2, mean_p2, std_p2 = aggregate_conv(all_conv_data_p2)
        it_sa, mean_sa, std_sa = aggregate_conv(all_conv_data_sa_rect)
        it_s, mean_s, std_s = aggregate_conv(all_conv_data_s_rect)

        # --- Determine overall max iterations and pad arrays ---
        len_p1 = len(it_p1) if it_p1 is not None else 0
        len_p2 = len(it_p2) if it_p2 is not None else 0
        len_sa = len(it_sa) if it_sa is not None else 0
        len_s = len(it_s) if it_s is not None else 0
        overall_max_len = max(len_p1, len_p2, len_sa, len_s)

        if overall_max_len > 0:
            # Pad P1 data
            if it_p1 is not None:
                padded_mean_p1 = np.pad(mean_p1, (0, overall_max_len - len_p1), 'edge')
                padded_std_p1 = np.pad(std_p1, (0, overall_max_len - len_p1), 'edge')
            else: # Handle case where P1 had no successful runs
                padded_mean_p1 = np.full(overall_max_len, np.nan)
                padded_std_p1 = np.full(overall_max_len, np.nan)

            # Pad P2 data
            if it_p2 is not None:
                padded_mean_p2 = np.pad(mean_p2, (0, overall_max_len - len_p2), 'edge')
                padded_std_p2 = np.pad(std_p2, (0, overall_max_len - len_p2), 'edge')
            else: # Handle case where P2 had no successful runs
                padded_mean_p2 = np.full(overall_max_len, np.nan)
                padded_std_p2 = np.full(overall_max_len, np.nan)

            # Pad SA-Rect data
            if it_sa is not None:
                padded_mean_sa = np.pad(mean_sa, (0, overall_max_len - len_sa), 'edge')
                padded_std_sa = np.pad(std_sa, (0, overall_max_len - len_sa), 'edge')
            else:
                padded_mean_sa = np.full(overall_max_len, np.nan)
                padded_std_sa = np.full(overall_max_len, np.nan)

            # Pad S-Rect data
            if it_s is not None:
                padded_mean_s = np.pad(mean_s, (0, overall_max_len - len_s), 'edge')
                padded_std_s = np.pad(std_s, (0, overall_max_len - len_s), 'edge')
            else:
                padded_mean_s = np.full(overall_max_len, np.nan)
                padded_std_s = np.full(overall_max_len, np.nan)


            it_common = np.arange(overall_max_len)

            plt.figure(figsize=(10, 6))
            
            # Plot P1 if it ran
            if it_p1 is not None:
                 plt.fill_between(it_common, padded_mean_p1 - padded_std_p1, padded_mean_p1 + padded_std_p1, alpha=0.2, color='tab:blue')
                 plt.plot(it_common, padded_mean_p1, color='tab:blue', linestyle='-', marker='.', markersize=3, label='Ours (Non-Rect, mean ± 1σ)')

            # Plot P2 if it ran
            if it_p2 is not None:
                 plt.fill_between(it_common, padded_mean_p2 - padded_std_p2, padded_mean_p2 + padded_std_p2, alpha=0.2, color='tab:orange')
                 plt.plot(it_common, padded_mean_p2, color='tab:orange', linestyle='--', marker='x', markersize=3, label='CPI (Non-Rect, mean ± 1σ)')

            # Plot SA-Rect if it ran
            if it_sa is not None:
                 plt.fill_between(it_common, padded_mean_sa - padded_std_sa, padded_mean_sa + padded_std_sa, alpha=0.2, color='tab:green')
                 plt.plot(it_common, padded_mean_sa, color='tab:green', linestyle=':', marker='s', markersize=3, label='SA-Rect L2 (mean ± 1σ)')

            # Plot S-Rect if it ran
            if it_s is not None:
                 plt.fill_between(it_common, padded_mean_s - padded_std_s, padded_mean_s + padded_std_s, alpha=0.2, color='tab:red')
                 plt.plot(it_common, padded_mean_s, color='tab:red', linestyle='-.', marker='^', markersize=3, label='S-Rect L2 (mean ± 1σ)')


            if valid_benchmarks:
                 plt.axhline(np.mean(valid_benchmarks), color='r', linestyle=':', linewidth=2, label=f'Benchmark Min (avg = {np.mean(valid_benchmarks):.4f})')

            plt.xlabel('Iteration')
            plt.ylabel('Estimated Robust Value (mean across trials)')
            plt.title(f'{mdp_type.capitalize()} Convergence Comparison (S={S}, A={A}, beta={beta})\\n(Flat lines indicate converged value)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            # Optional: Zoom in on early iterations if needed
            # plt.xlim(0, min(overall_max_len, 50)) # Example: Zoom to first 50 iterations
            plt.tight_layout()
            plt.show()
        else:
             print("No convergence data available to plot error bands.")
        # =======================================================================

    if plot_time_comparison and len(data_times_all) >= 1:
        # Time Comparison Plot (Box Plot)
        plt.figure(figsize=(10, 6)) # Increased width for 4 boxes
        plt.boxplot(data_times_all, labels=plot_labels_all)
        plt.ylabel("Wall Clock Time (s)")
        plt.title(f"{mdp_type.capitalize()} Computation Time Comparison (S={S}, A={A}, beta={beta})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    if plot_gap_comparison and len(data_times_all) >= 1:
        # Gap Comparison Plot (Box Plot) - Use only valid gaps
        # Filter out any methods that had no valid gaps
        plot_data_gap_final = [g_list for g_list in data_gaps_all if g_list] # Ensure list is not empty
        plot_gap_labels_final = [plot_labels_all[i] for i, g_list in enumerate(data_gaps_all) if g_list]

        if plot_data_gap_final:
            plt.figure(figsize=(10, 6)) # Increased width for 4 boxes
            plt.boxplot(plot_data_gap_final, labels=plot_gap_labels_final)
            plt.ylabel("Suboptimality Gap (V_algo - V_benchmark_min)")
            plt.title(f"{mdp_type.capitalize()} Performance vs Benchmark (S={S}, A={A}, beta={beta})")
            # Adjust ylim carefully based on actual data
            min_gap_val = min(min(g_list, default=0) for g_list in plot_data_gap_final if g_list) if plot_data_gap_final else 0
            plt.ylim(bottom=-0.01 if min_gap_val > -0.01 else None)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid gap data to plot for any algorithm.")

    return results


# #############################################
# ####### Main Execution Block       ##########
# #############################################
if __name__ == "__main__":
    print("Script starting execution in main block...")

    # --- Common Parameters ---
    N_SAMPLES_OPT_single = 1000  # More samples for single MDP comparison (accuracy)
    N_SAMPLES_OPT_multi = 100    # Fewer samples for multi-trial analysis (speed)
    P1_TOL_common = 1e-6         # Tolerance for Paper 1 binary search
    P2_TOL_common = 1e-6         # Tolerance for Paper 2 FW gap
    CPI_MAX_ITER_common = 100    # Max iterations for CPI
    GAMMA_common = 0.9           # Discount factor
    N_TRIALS_performance = 20    # Number of trials for performance analysis

    # --- Experiment Control Flags ---
    RUN_PERFORMANCE_ANALYSIS = False  # Set to True to enable performance analysis

    print("\n" + "="*80)
    print("EXPERIMENT STRUCTURE:")
    print("1. Single MDP Comparison: Compare algorithms on the SAME MDP")
    print("2. Performance Analysis: Statistical comparison across multiple MDPs")
    print(f"   Performance Analysis: {'ENABLED' if RUN_PERFORMANCE_ANALYSIS else 'DISABLED'}")
    print("="*80)

    # =======================================================================
    # EXPERIMENT TYPE 1: SINGLE MDP COMPARISONS
    # =======================================================================
    print("\n" + "="*60)
    print("EXPERIMENT TYPE 1: SINGLE MDP COMPARISONS")
    print("Compare all 4 algorithms on the SAME MDP instance")
    print("="*60)

    # Single MDP Comparison Set 1: Varying State Space (A=10, beta=0.01)
    print("\n" + "-"*50)
    print("Single MDP Comparison Set 1: Varying State Space")
    print("A=10, β=0.01, S=10,50,100,200")
    print("-"*50)
    
    S_values_set1 = [10, 50, 100, 200]
    for S_val in S_values_set1:
        print(f"\n{'='*15} Running S={S_val}, A=10, β=0.01 {'='*15}")
        run_single_mdp_comparison(S=S_val, A=10, beta=0.01, gamma=GAMMA_common,
                                 n_samples_opt=N_SAMPLES_OPT_single,
                                 p1_tol=P1_TOL_common, p2_tol=P2_TOL_common, 
                                 cpi_max_iter=CPI_MAX_ITER_common,
                                 mdp_type="random", plot_convergence=True)

    # Single MDP Comparison Set 2: Varying Beta (A=10, S=100)
    print("\n" + "-"*50)
    print("Single MDP Comparison Set 2: Varying Beta")
    print("A=10, S=100, β=0.005,0.01,0.05,0.1")
    print("-"*50)
    
    beta_values_set2 = [0.005, 0.01, 0.05, 0.1]
    for beta_val in beta_values_set2:
        print(f"\n{'='*15} Running S=100, A=10, β={beta_val} {'='*15}")
        run_single_mdp_comparison(S=100, A=10, beta=beta_val, gamma=GAMMA_common,
                                 n_samples_opt=N_SAMPLES_OPT_single,
                                 p1_tol=P1_TOL_common, p2_tol=P2_TOL_common, 
                                 cpi_max_iter=CPI_MAX_ITER_common,
                                 mdp_type="random", plot_convergence=True)

    # =======================================================================
    # EXPERIMENT TYPE 2: PERFORMANCE ANALYSIS (OPTIONAL)
    # =======================================================================
    if RUN_PERFORMANCE_ANALYSIS:
        print("\n" + "="*60)
        print("EXPERIMENT TYPE 2: PERFORMANCE ANALYSIS")
        print("Statistical comparison across multiple random MDP instances")
        print("="*60)

        # Performance Analysis 1: Varying State Space Size
        print("\n" + "-"*50)
        print("Performance Analysis 1: Varying State Space Size")
        print("-"*50)
        
        S_values = [50, 100, 150, 200]
        for S_val in S_values:
            print(f"\n{'='*15} Performance Analysis: S={S_val} {'='*15}")
            run_performance_analysis(S=S_val, A=10, beta=0.01, gamma=GAMMA_common,
                                    n_trials=N_TRIALS_performance, 
                                    n_samples_opt=N_SAMPLES_OPT_multi,
                                    p1_tol=P1_TOL_common, p2_tol=P2_TOL_common, 
                                    cpi_max_iter=CPI_MAX_ITER_common,
                                    mdp_type="random")

        # Performance Analysis 2: Varying Beta (Uncertainty Radius)
        print("\n" + "-"*50)
        print("Performance Analysis 2: Varying Beta (Uncertainty Radius)")
        print("-"*50)
        
        beta_values = [0.005, 0.01, 0.02, 0.05]
        for beta_val in beta_values:
            print(f"\n{'='*15} Performance Analysis: β={beta_val} {'='*15}")
            run_performance_analysis(S=100, A=10, beta=beta_val, gamma=GAMMA_common,
                                    n_trials=N_TRIALS_performance, 
                                    n_samples_opt=N_SAMPLES_OPT_multi,
                                    p1_tol=P1_TOL_common, p2_tol=P2_TOL_common, 
                                    cpi_max_iter=CPI_MAX_ITER_common,
                                    mdp_type="random")
    else:
        print("\n" + "="*60)
        print("EXPERIMENT TYPE 2: PERFORMANCE ANALYSIS - SKIPPED")
        print("Set RUN_PERFORMANCE_ANALYSIS = True to enable")
        print("="*60)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED!")
    print(f"\nSUMMARY:")
    print("✅ Single MDP comparisons completed for:")
    print("   - Varying S: [10, 50, 100, 200] with A=10, β=0.01")
    print("   - Varying β: [0.005, 0.01, 0.05, 0.1] with S=100, A=10")
    if RUN_PERFORMANCE_ANALYSIS:
        print("✅ Performance analysis completed")
    else:
        print("⏭️  Performance analysis skipped (RUN_PERFORMANCE_ANALYSIS = False)")
    print("="*80)
