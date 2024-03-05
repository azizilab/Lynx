import torch


# Channel-wise mean & std
CYIF_MEAN = torch.tensor([
    0.1672,         # DAPI
    0.0033,         # GS 
    0.1008,         # CYP
    0.0331,         # ASS
    0.0427,         # CD31
    0.1700,         # CD45
    0.2084,         # CD68
    0.0426          # CD56
])

CYIF_STD = torch.tensor([
    0.1106,         # DAPI
    0.0207,         # GS 
    0.0517,         # CYP
    0.0649,         # ASS
    0.0070,         # CD31
    0.0101,         # CD45
    0.0026,         # CD68
    0.0075          # CD56
])


HE_MEAN = torch.tensor([
    0.6608,         # R
    0.4802,         # G 
    0.6315,         # B
])

HE_STD = torch.tensor([
    0.2144,         # R
    0.2288,         # G
    0.1734,         # B
])
