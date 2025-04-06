import numpy as np

THRESHOLD = 1e-8  # Threshold to determine if a value is close to zero

def compute_formula(s, k, func_type='cos'):
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
    # In fact, since the parameters $\varphi_k$ and $\psi_k$ in Equation (15) of the appendix are unrelated to our previous derivations,
    # they can be chosen arbitrarily. In our calculation,,
    # we set $\varphi_k = 0$ for sine components and $\psi_k = \frac{\pi}{2}$ for cosine components.
    if func_type == 'cos':
        values = np.cos(angles + np.pi / 2)
    elif func_type == 'sin':
        values = np.sin(angles)

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
    for s in range(20, 512):
        print(f"Processing s={s}...")
        for k in range(1, s + 1):
            result_cos = compute_formula(s, k, func_type='cos')
            result_sin = compute_formula(s, k, func_type='sin')
            # Skip printing if both results are zero
            if result_cos == 0 and result_sin == 0:
                continue
            results_found = True
            print(f"s={s}, k={k} => cos: {result_cos}, sin: {result_sin}")

    if not results_found:
        print("All calculated results are zero")