# constraints_library/full_rules.py

"""
This file contains all registered spatial constraint functions.
Each function should take (token, sequence) as input and return a boolean.
New functions are appended here automatically when a user registers new spatial constraints.
"""

def no_duplicate(token, sequence):
    """
    Constraint to ensure that a token does not already exist in the sequence.
    """
    return token not in sequence

def within_distance(token, sequence, max_distance=0.01):
    """
    Constraint to ensure that the token is within a maximum distance from the last token in the sequence.
    This assumes token and sequence elements are (lat, lon) tuples.
    """
    from math import sqrt

    if not sequence:
        return True

    last_token = sequence[-1]
    lat_diff = token[0] - last_token[0]
    lon_diff = token[1] - last_token[1]
    distance = sqrt(lat_diff**2 + lon_diff**2)

    return distance <= max_distance

# === DO NOT REMOVE THIS LINE ===
# New constraint functions will be appended below automatically.