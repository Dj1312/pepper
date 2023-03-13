from functools import cached_property
from typing import List, Tuple

import pydantic as pd
import numpy as np

# from tidy3d.components.medium import AbstractMedium, DispersiveMedium, ensure_freq_in_range
# from tidy3d.components.base import Tidy3dBaseModel
# from tidy3d.constants import C_0, MICROMETER
from tidy3d.components.medium import PoleResidue
from tidy3d.material_library.material_library import MaterialItem, VariantItem, material_library
from tidy3d.material_library.material_reference import material_refs, ReferenceData


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


# class LithiumNiobate = td.VariantItem()
# class VariantItem(Tidy3dBaseModel):
#     """Reference, data_source, and material model for a variant of a material."""

#     medium: PoleResidue = pd.Field(
#         ...,
#         title="Material dispersion model",
#         description="A dispersive medium described by the pole-residue pair model.",
#     )

#     reference: List[ReferenceData] = pd.Field(
#         None,
#         title="Reference information",
#         description="A list of reference related to this variant model.",
#     )

#     data_url: str = pd.Field(
#         None,
#         title="Dispersion data URL",
#         description="The URL to access the dispersion data upon which the material "
#         "model is fitted.",
#     )


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
    data_url="https://www.allresist.com/wp-content/uploads/sites/2/2020/03/AR-P6200_CSAR62english_Allresist_product-information.pdf",
)

material_library['Csar62'] = MaterialItem(
    name="CSAR62 resist",
    variants=dict(
        AllResist=Csar62
    ),
    default="AllResist",
)
