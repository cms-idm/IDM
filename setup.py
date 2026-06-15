"""Simple setup module to allow for pip installation of the idm analysis package."""

import setuptools

setuptools.setup(
    name="idm",
    version="0.1.0",
    description="Inelastic Dark Matter (Run 3, e+mu) analysis package",
    url="https://github.com/cms-idm/IDM",
    license="BSD 3-clause",
    packages=["idm", "idm.tools", "idm.definitions"],
    include_package_data=True,
)
