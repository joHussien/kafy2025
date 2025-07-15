import random

def apply_spatial_rules(spatial_rules, current_traj, candidate_tokens):
    """
    Apply spatial constraints on candidate tokens.
    This is a placeholder — the actual logic should use the real rules.
    """
    # For now, return all candidates assuming all are valid
    # You can apply real spatial logic using spatial_rules later
    return candidate_tokens

def run_operation(model, tokenized_trajectory, spatial_rules, **kwargs):
    """
    Logic for trajectory generation.

    Args:
        model: The loaded model instance.
        tokenized_trajectory: Input seed trajectory (ignored in this pseudocode).
        spatial_rules: List of spatial rule functions to apply.
        **kwargs: Additional parameters (MaxLength, Count, MaxTrials, etc.)

    Returns:
        List of generated trajectories.
    """

    max_len = kwargs["MaxLength"]
    count = kwargs["Count"]
    max_trials = kwargs.get("MaxTrials", 100)
    return
    generated_trajectories = []

    for _ in range(count):
        start_token = model(None)  # generate first token
        current_traj = [start_token]
        i = 0

        while len(current_traj) < max_len and i < max_trials:
            candidate_tokens = model(current_traj)  # e.g. next-token distribution
            valid_tokens = apply_spatial_rules(spatial_rules, current_traj, candidate_tokens)

            if not valid_tokens:
                break  # No valid next step

            next_token = random.choice(valid_tokens)  # Or use top-1, top-k, etc.
            current_traj.append(next_token)
            i += 1

        generated_trajectories.append(current_traj)

    return generated_trajectories