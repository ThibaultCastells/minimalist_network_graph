from setuptools import setup, find_packages

setup(
    name='minimalist-network-graph',
    version='0.0.1',
    description='A tool to get a minimalist view of any architecture.',
    url='https://github.com/ThibaultCastells/minimalist_network_graph',
    author='Thibault Castells',
    packages=find_packages(exclude=['docs', 'demo', 'results']),
)
