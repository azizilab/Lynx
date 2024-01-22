import torch


# Channel-wise mean & std
CYIF_MEAN = torch.tensor([
    0.0050,         # GS 
    0.1015,         # CYP
    0.0425,         # ASS
])

CYIF_STD = torch.tensor([
    0.0237,         # GS
    0.0532,         # CYP
    0.0704,         # ASS
])