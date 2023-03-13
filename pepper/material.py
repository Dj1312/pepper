from functools import cached_property
from typing import List, Tuple

import pydantic as pd
import numpy as np

# from tidy3d.components.medium import AbstractMedium, DispersiveMedium, ensure_freq_in_range
# from tidy3d.components.base import Tidy3dBaseModel
# from tidy3d.constants import C_0, MICROMETER
from tidy3d.components.medium import PoleResidue, Sellmeier
from tidy3d.material_library.material_library import MaterialItem, VariantItem
from tidy3d.material_library.material_library import material_library as tidy3d_material_library
from tidy3d.material_library.material_reference import ReferenceData
from tidy3d.material_library.material_reference import material_refs as tidy3d_material_refs


# Impossible to use actually since Tidy3D want a PoleResidue model
# class Cauchy(DispersiveMedium):
#     """A dispersive medium described by the Cauchy model.
#     The frequency-dependence of the refractive index is described by:
#     Note
#     ----
#     .. math::
#         n(\\lambda) = \\sum_i \\frac{N_i}{\\lambda^{i * 2N}}
#     Example
#     -------
#     >>> cauchy_medium = Cauchy(coeffs=[(1,2)])
#     >>> eps = cauchy_medium.eps_model(200e12)
#     """

#     coeffs: Tuple[float, pd.PositiveFloat] = pd.Field(
#         title="Coefficients",
#         description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
#         units=(None, MICROMETER + "^2", MICROMETER + "^4", ...),
#     )

#     def _n_model(self, frequency: float) -> complex:
#         """Complex-valued refractive index as a function of frequency."""

#         wvl = C_0 / np.array(frequency)
#         wvl2 = wvl**2
#         n = 0.0
#         for idx, N in enumerate(self.coeffs):
#             n += N / wvl2**idx
#         return n

#     @ensure_freq_in_range
#     def eps_model(self, frequency: float) -> complex:
#         """Complex-valued permittivity as a function of frequency."""

#         n = self._n_model(frequency)
#         return AbstractMedium.nk_to_eps_complex(n)

new_material_refs = dict(
    AllResist=ReferenceData(
        journal="E-Beam Resist AR-P 6200 series (CSAR 62) Product information",
        url="https://www.allresist.com/wp-content/uploads/sites/2/2020/03/AR-P6200_CSAR62english_Allresist_product-information.pdf",
    ),
    Zelmon1997=ReferenceData(
        journal="D. E. Zelmon, D. L. Small, and D. Jundt. "
        "Infrared corrected Sellmeier coefficients for congruently grown "
        "lithium niobate and 5 mol.% magnesium oxide-doped lithium niobate, "
        "J. Opt. Soc. Am. B 14, 3319-3322 (1997)",
        doi="https://doi.org/10.1364/JOSAB.14.003319",
    ),
)


Csar62 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            (
                (1.3727107194260728e+17j),
                (-6.093034981754017e+16j)
            ),
            (
                (-1.83658298289027e+17j),
                (4.528208825831985e+16j)
            ),
        ),
        frequency_range=(119964969187675.06, 599584916000000.0)),
    reference=[new_material_refs["AllResist"]],
    data_url="https://www.allresist.com/wp-content/uploads/sites/2/2020/03/AR-P6200_CSAR62english_Allresist_product-information.pdf",
)


LiNbO3_o = VariantItem(
    medium=Sellmeier(
        coeffs=(
            (
                (2.6734),
                (0.01764)
            ),
            (
                (1.2290),
                (0.05914)
            ),
            (
                (12.614),
                (474.60)
            ),
        ),
        frequency_range=(59958491600000.0, 749481145000000.0)
    ).pole_residue,
    reference=[new_material_refs["Zelmon1997"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=data/main/LiNbO3/Zelmon-o.yml",
)


LiNbO3_e = VariantItem(
    medium=Sellmeier(
        coeffs=(
            (
                (2.9804),
                (0.02047)
            ),
            (
                (0.5981),
                (0.0666)
            ),
            (
                (8.9543),
                (416.08)
            ),
        ),
        frequency_range=(59958491600000.0, 749481145000000.0),
    ).pole_residue,
    reference=[new_material_refs["Zelmon1997"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=data/main/LiNbO3/Zelmon-e.yml",
)


new_material_library = dict(
    Csar62=MaterialItem(
        name="CSAR62 resist",
        variants=dict(
            AllResist=Csar62
        ),
        default="AllResist",
    ),
    LiNbO3=MaterialItem(
        name="Lithium Niobate",
        variants=dict(
            ordinary=LiNbO3_o,
            extraordinary=LiNbO3_e,
        ),
        default="ordinary"
    )
)


# PEP 584 add the possibility to merge dict using | operator,
# but since it is valable for Python>=3.9, use this method:
material_refs = {**tidy3d_material_refs, **new_material_refs}
material_library = {**tidy3d_material_library, **new_material_library}
