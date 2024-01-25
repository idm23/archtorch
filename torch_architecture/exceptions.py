"""File defining exceptions for TorchArchitecture
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
# ==================================

# ========= program imports ========
# ==================================

class ModelConstructionError(Exception):
    pass

class ModelSizingError(ModelConstructionError):
    pass

class DataRetrievalError(Exception):
    pass