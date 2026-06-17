from setuptools import setup, find_packages

setup(
    name='uafgi',
    version='0.0.1',
    url='https://github.com/pism/uafgi',
    author='Elizabeth Fischer',
    author_email='eafischer2@alaska.edu',
    description='Some tools for internal use in in research at UAF/GI',
    packages=find_packages(),    
    python_requires='>=3.8',
)