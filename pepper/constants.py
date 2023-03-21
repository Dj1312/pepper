from math import sqrt, pi

"""Principal constants used in Pepper for the mode solving"""

# General physics constants - From NIST Reference
PI = pi
EPS_0 = 8.85418782e-12  # Vacuum permittivity           [F.m-1]
MU_0 = 4 * PI * 1e-7  # Vacuum permeability             [H.m-1] = [N.A-2]
C_0 = sqrt(1 / (EPS_0 * MU_0))  # Vacuum speed of light [m.s-1]
ETA_0 = sqrt(MU_0 / EPS_0)  # Impedance of free space   [Ohm]

# Parameters used for the PML calculations - From Taflove 4th edition
FACTOR_M_FDFD = 3#4
FACTOR_R_FDFD = -30#-12  # Factor corresponding to R(0) in Taflove's book

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge

# Length values
MICROMETERS = 1e-6
