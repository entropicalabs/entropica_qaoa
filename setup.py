try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


setup(
    name="forest_qaoa",
    version="0.1dev",
    description="QAOA implementation on top of pyQuil",
    author="Entropica Labs: Jan Lukas Bosse, Ewan Munro",
    packages=find_packages(),
    install_requires=['numpy >= 1.7', 'scipy >= 0.9',
                      'scikit-learn >= 0.16', 'numexpr >= 2.5']
)