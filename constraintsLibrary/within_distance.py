def apply_constraint(token, sequence, max_distance=0.01):
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