import numpy as np
from tqdm import tqdm
from fractions import Fraction
from functools import reduce
from math import gcd
import re


def min_max_normalize(distance_matrix):
    """
    Normalize a distance matrix using Min-Max scaling to the range [0, 1].

    Args:
        distance_matrix (numpy.ndarray): Square distance matrix to be normalized

    Returns:
        numpy.ndarray: Normalized distance matrix with values scaled to [0,1]
    """
    min_val = np.min(distance_matrix)
    max_val = np.max(distance_matrix)
    return (distance_matrix - min_val) / (max_val - min_val)


def drop_diagonal(distance_matrix):
    """
    Remove diagonal elements from a square distance matrix.

    Args:
        distance_matrix (numpy.ndarray): Square distance matrix

    Returns:
        numpy.ndarray: Distance matrix with diagonal elements removed
    """
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    distance_matrix = distance_matrix[mask].reshape(distance_matrix.shape[0], -1)
    return distance_matrix


def is_pareto_efficient(points, return_mask=True):
    """
    Find Pareto-efficient points in a set of points (assuming minimization).
    A point is Pareto-efficient if no other point dominates it in all dimensions.

    Args:
        points (numpy.ndarray): Array of shape (n_points, n_dimensions) containing points
        return_mask (bool): If True, return boolean mask, if False return indices

    Returns:
        numpy.ndarray: Boolean mask or indices array indicating Pareto-efficient points
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if is_efficient[i]:
            is_efficient[is_efficient] = ~np.all(
                points[i] <= points[is_efficient],
                axis=1,
            )
            is_efficient[i] = True

    if return_mask:
        return is_efficient
    return np.where(is_efficient)[0]


def find_last_element_position(formula, max_length):
    """
    Find the position of the last complete chemical element within a maximum length.
    Used for truncating chemical formulas at element boundaries.

    Args:
        formula (str): Chemical formula string
        max_length (int): Maximum allowed length for truncation

    Returns:
        int: Position of the last complete element ending before max_length
    """
    element_positions = [
        m.start() for m in re.finditer(r"[A-Z][a-z]?(?:\$_{[^}]*}\$)?", formula)
    ]

    for pos in reversed(element_positions):
        if pos <= max_length:
            match = re.match(r"[A-Z][a-z]?(?:\$_{[^}]*}\$)?", formula[pos:])
            if match:
                return pos + len(match.group(0))
    return max_length


def format_to_latex(formula):
    """
    Convert a chemical formula to LaTeX format with proper subscripts.
    Handles truncation of long formulas at element boundaries.

    Args:
        formula (str): Chemical formula string to be formatted

    Returns:
        str: LaTeX formatted chemical formula with subscripts
    """
    pattern = r"([A-Za-z])([0-9]+\.?[0-9]*)"
    formatted = re.sub(pattern, r"\1$_{\2}$", formula)

    max_length = 50
    if len(formatted) > max_length:
        end_pos = find_last_element_position(formatted, max_length)
        formatted = formatted[:end_pos] + "$_{..}$"

    return formatted


def reduce_single_formula(formula):
    """
    Reduce a chemical formula to its lowest ratio with proper formatting.
    Handles large numbers by allowing decimal values and converts to LaTeX format.

    Args:
        formula (str): Chemical formula string to be reduced

    Returns:
        str: Reduced and LaTeX formatted chemical formula
    """
    parts = formula.split()
    numbers = []
    elements = []

    for part in parts:
        element_match = re.match(r"([A-Z][a-z]*)", part)
        number_match = re.search(r"([0-9]+\.?[0-9]*)", part)

        if element_match:
            elements.append(element_match.group(1))
        if number_match:
            numbers.append(float(number_match.group(1)))
        else:
            numbers.append(1.0)

    if not numbers:
        return formula.replace(" ", "")

    fractions = [Fraction(str(n)).limit_denominator() for n in numbers]
    numerators = [f.numerator for f in fractions]
    denominators = [f.denominator for f in fractions]

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    lcm_value = reduce(lcm, denominators)
    scaled_numbers = [(n * lcm_value) // d for n, d in zip(numerators, denominators)]
    gcd_value = reduce(gcd, scaled_numbers)

    reduced_numbers = [n / gcd_value for n in scaled_numbers]

    max_reduced = max(abs(n) for n in reduced_numbers)
    if max_reduced > 30:
        factor = max_reduced / 30
        reduced_numbers = [n / factor for n in reduced_numbers]

    reduced_numbers = [round(n, 2) for n in reduced_numbers]

    result = ""
    for element, number in zip(elements, reduced_numbers):
        if number == 1:
            result += element
        else:
            result += f"{element}{number:.2f}".rstrip("0").rstrip(".")

    result = re.sub(r"([A-Za-z])\.([0-9]+)", r"\1\2", result)

    formatted = format_to_latex(result)
    formatted = re.sub(r"_\{\.([0-9]+)\}", r"_{0.\1}", formatted)

    return formatted
