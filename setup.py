from setuptools import setup, find_packages

setup(
    name='mosaictools',
    version='0.1.0',
    author='Blaž Kurent',
    author_email='blaz.kurent@fgg.uni-lj.si',
    description='Surrogate modelling of modal properties using MOSAIC method',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/blazkurent/mosaictools/',
    packages=find_packages(),
    py_modules=['mosaictools'],
    install_requires=[
        'numpy',
        'sklearn',
        'scipy',
        'copy',
        'abc'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)
