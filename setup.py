from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name='flux',
    version='0.1.0',
    description='Portable Data Science Pipelines',
    long_description=long_description,
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
     #external packages as dependencies
    install_requires=[
        'pandas', 
        # 'numpy', 
        # 'scikit-learn', 
        'dill', 
        'PyYAML'
    ],
),