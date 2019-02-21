from setuptools import setup, find_packages

setup(
    name='qubogen',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/tamuhey/qubogen',
    install_requires=["networkx"],
    license='MIT',
    author='tamuhey',
    author_email='tamuhey@gmail.com',
    description='QUBO matrix generator on major combinatorial optimization problems written in Python',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
