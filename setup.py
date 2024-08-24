from setuptools import find_packages, setup
from typing import List

IGNORE = '-e .'

def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path, 'r') as f:
        requirements=f.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if IGNORE in requirements:
            requirements.remove(IGNORE)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Rishika',
    author_email='rishiagrawal2311@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
)