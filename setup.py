from setuptools import setup, find_packages

# Set up file so the repo can be used as a pacakge

setup(
    name='mma_model',  # This becomes your package name
    version='0.1.2',
    packages=['functions'],
    package_dir={'functions': 'functions'},
    description='Reusable functions from MMA model',
    author='Andy Blinston',
    url='https://github.com/ablinston/mma_model_lite',
)