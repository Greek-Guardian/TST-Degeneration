import numpy as np

THRESHOLD = 1e-8  # Threshold to determine if a value is zero

def compute_formula(s, k, theta, func_type='cos'):
    """
    Parameters:
        s: positive integer, determines the summing range up to 2 * s
        k: k parameter
        func_type: type of trigonometric function to use ('cos' or 'sin')
    Returns:
        The computed result of the formula.
    """
    if func_type == 'sin' and k == s:
        return 0.0

    # Create an array from 1 to 2 * s
    indices = np.arange(1, 2 * s + 1)

    # Compute each angle
    angles = 2 * np.pi * k * (indices / (2 * s))

    # Compute values
    if func_type == 'cos':
        values = np.cos(angles + theta)
    elif func_type == 'sin':
        values = np.sin(angles + theta)

    # If a value is within the threshold of 0, set it to 0; otherwise, use its sign
    signed_values = np.where(np.abs(values) <= THRESHOLD, 0.0, np.sign(values))

    # Sum the values and return the result
    return np.sum(signed_values)

if __name__ == "__main__":

    results_found = False  # Flag to determine if any non-zero results are produced

    # Check for s in the range of 20 to 512
    # and k in the range of 1 to s
    # and print the results
    # If both results are zero, skip printing
    # If all results are zero, print a message
    # for s in range(20, 257):
    for s in range(256, 257):
        print(f"Processing n={2*s}, s={s}...")
        fixed_theta_num = 0
        for k in range(1, s + 1):
            theta_cos = np.random.uniform(0, 2 * np.pi)
            theta_sin = np.random.uniform(0, 2 * np.pi)
            result_cos = compute_formula(s, k, theta_cos, func_type='cos')
            result_sin = compute_formula(s, k, theta_sin, func_type='sin')
            # In fact, since the parameters $\varphi_k$ and $\psi_k$ in Equation (15) of the appendix are unrelated to our previous derivations, they can be chosen arbitrarily.
            # In our calculation, if necessary, we will set $\varphi_k = 0$ for sine components and $\psi_k = \frac{\pi}{2}$ for cosine components.
            if result_cos != 0:
                fixed_theta_num += 1
                result_cos = compute_formula(s, k, np.pi / 2, func_type='cos')
            if result_sin != 0:
                fixed_theta_num += 1
                result_sin = compute_formula(s, k, 0, func_type='sin')
            # Skip printing if both results are zero
            if result_cos == 0 and result_sin == 0:
                continue
            results_found = True
            print(f"s={s}, k={k} => cos: {result_cos}, sin: {result_sin}")
        # print(f"Fixed theta count for n={2*s}: {fixed_theta_num}\n")

    if not results_found:
        print("All calculated results are zero")