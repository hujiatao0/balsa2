import balsa


def SanitizeToText(d):
    """Sanitizes result/config dicts for W&B textual logging."""
    # For configs: flattens all so w&b can visualize nicely.
    # Don't quote all because otherwise scalars are converted to strings.
    ret, _ = balsa.hyperparams.ToFlattenedTextDict(d, quote_all=False)
    return ret