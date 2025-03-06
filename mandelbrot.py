import numpy as np


def get_escape_time(c:complex, max_iterations: int) -> int | None:
    """Returns int - the number of iterations which pass before c escapes -
      or None - if c does not escape (< 2) in the specified number of iterations"""
    number_of_iterations = 0
    real = c.real
    imaginary = c.imag
    for num in range(max_iterations):
        expression = abs(real + (imaginary * num))
        if expression == 2:
            return number_of_iterations
        elif real + abs(imaginary * (num+1)) > 2:
            return number_of_iterations
        elif expression < 2:
           number_of_iterations += 1
    return None
def get_escape_time_color_arr(
        c_arr: np.ndarray,
        max_iterations: int
) -> np.ndarray:
    a = np.zeros_like(c_arr)
    