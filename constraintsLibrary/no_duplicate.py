def apply_constraint(token, sequence):
    """
    Constraint to ensure that a token does not already exist in the sequence.
    """
    return token not in sequence