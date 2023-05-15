# 🌶️ Pepper - Work In Progress...

Pepper is born from the idea to wrap [Tidy3D](https://github.com/flexcompute/tidy3d)
for [MEEP](https://github.com/NanoComp/meep).
Actually, it was more of a playground to use basic FDFD from designs made by Tidy3D.

In the future, the FDFD will be extended to be fully JAX compatible,
to compute S-parameters or diffraction orders, and more! Pepper also aims to offer a state-of-the-art CUDA-based 3D FDFD.

The goal of the package will be to connect the ease and the power of Tidy3D with
multiple backends such as [MEEP](https://github.com/NanoComp/meep), [fdtdz](https://github.com/spinsphotonics/fdtdz), ..., to allow the user to compute and optimize devices
by taking the best advantages of numerous different methods.


## Credits
FDFD is based on the work of Yu Jerry Shi on [fdfd_suite](https://github.com/YuJerryShi/fdfd_suite)
which was also used on the [Ceviche](https://github.com/fancompute/ceviche) package.