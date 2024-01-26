"""Setup script for archtorch
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
# ==================================

# ========= program imports ========
import setuptools
# ==================================

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='archtorch',
    version='0.9.0',
    description=long_description,
    author='Ian Mackey',
    author_email='idm@ianmackey.net',
    packages=setuptools.find_packages(include=['archtorch', 'archtorch*']),
    python_requires='<3.12',
    install_requires=[
        'torch>=2.1.0', # There are issues with this. Torch is required but installation isn't so simple.
        'numpy>=1.26.0',
        'pytest'
    ],
)