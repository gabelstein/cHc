from setuptools import setup, find_packages

setup(
    name='cHc',
    version='0.1',
    packages=["chc"],
    url='https://github.com/gabelstein/cHc',
    license='',
    author='Gabriel',
    author_email='gabriel@bccn-berlin.de',
    description='a convex hull classifier',
    install_requires=['numpy', 'scipy', 'scikit-learn']
)
