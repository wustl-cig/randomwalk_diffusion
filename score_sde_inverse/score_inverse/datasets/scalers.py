def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config['imagedetail']['centered']:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config['imagedetail']['centered']:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x
