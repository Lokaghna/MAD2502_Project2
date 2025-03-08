import numpy as np


def get_escape_time(c:complex, max_iterations: int) -> int | None:
    """Returns int - the number of iterations which pass before c escapes -
        or None - if c does not escape (< 2) in the specified number of iterations"""

    z = 0

    for num in range(max_iterations):
        sequence = z = (z * z) + c
        if sequence == 2:
            return num
        elif abs(sequence) > 2:
            return num

    return None


def get_complex_grid(top_left: complex, bottom_right: complex, step: float) -> np.ndarray:
    """Compute a grid of complex numbers, one for each pixel of our image;
        returns array - contents will be complex numbers evenly spaced between top_left
        and (but not including) bottom_right"""

    real_num = np.arange(top_left.real, bottom_right.real, step) #found the attributes - real and imag - using dir(np)
    imag_num = np.arange(top_left.imag, bottom_right.imag, -step)

    reshape_real = np.reshape(real_num, (1, len(real_num)))
    reshape_imag = np.reshape(imag_num, (1, len(imag_num), 1))

    final_grid = reshape_real + (1j * reshape_imag)

    return final_grid

def get_escape_time_color_arr(
        c_arr: np.ndarray,
        max_iterations: int
) -> np.ndarray:
    a = np.zeros_like(c_arr)
    