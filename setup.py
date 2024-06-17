from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent


long_description = (this_directory / ".pip_readme.rst").read_text()
requirements = (
    (this_directory / "requirements" / "requirements-core.txt").read_text().split("\n")
)


setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    name="s2scat",
    version="0.0.1",
    url="https://github.com/astro-informatics/s2scat",
    author="Matthew A. Price, Louise Mousset, Erwan Allys, Jason D. McEwen",
    license="MIT",
    python_requires=">=3.8",
    install_requires=requirements,
    description=(
        "Differentiable and GPU accelerated scattering covariance statistics on the sphere"
    ),
    long_description_content_type="text/x-rst",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    pacakge_data={"s2scat": ["default-logging-config.yaml"]},
)
