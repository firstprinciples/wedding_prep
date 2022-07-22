import setuptools
import apeelx_wedding_prep

# Get the version of this package
version = apeelx_wedding_prep.version

# Get the long description of this package
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='apeelx_wedding_prep',
    version=version,
    author="Apeel Data Science",
    author_email="data.science@apeelsciences.com",
    description="A short description of the project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/apeelsciences/datascience/exploratory/wedding_prep",
    packages=setuptools.find_packages(exclude=['unit_tests']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'Pillow',
        'scikit-learn',
        'scikit-image',
        'networkx',
        'seaborn',
        'tqdm',
        'plotly',
        'gurobipy',
        'nbformat'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
