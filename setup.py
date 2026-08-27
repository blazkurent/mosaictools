from setuptools import setup, find_packages

setup(
    name='mosaictools',
    version='0.2.2',
    author='Blaž Kurent, Bence Popovics',
    author_email='blaz.kurent@fgg.uni-lj.si, popbence@hun-ren.sztaki.hu',
    description='Surrogate modelling of modal properties using MOSAIC method',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/blazkurent/mosaictools/',
    packages=find_packages(exclude=["mosaic_example_data", "mosaic_example_data.*", "notebooks", "notebooks.*"]),
    py_modules=['mosaictools'],
    install_requires=[
        'numpy<2.0',
        'scikit-learn',
        'scipy',
        'uncertain-variables',
        'gPCE-model',
        'pandas',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)
