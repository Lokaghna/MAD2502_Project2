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
    """Compute the escape of the color array for a given grid of complex numbers based on an iteration process
    Return:
    --------
    An array that has the same shape with the c_arr, containing values in the range[0,1] so that:
    -Points that never escape are colored black(0.0)
    -Points with 0 escape time are colored white(1.0)
    -Points with maximum escape time are colored with 1/(max_iterations+1), which is close to 0.0"""
    a = np.zeros_like(c_arr)
    escape_time = np.full(c_arr.shape, max_iterations + 1)
    escape_points = np.ones(c_arr.shape, dtype=bool)
    for n in range(max_iterations):
        a[escape_points] = a[escape_points] * a[escape_points] + c_arr[escape_points]
        escaped = np.abs(a) > 2
        escape_time[escaped & escape_points] = n
        escape_points[escaped] = False

    color = (max_iterations - escape_time +1 )/ (max_iterations + 1)
    color[escape_time == 0] = 1.0
    return color.reshape(color.shape[1:])





    