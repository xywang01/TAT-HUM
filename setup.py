from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='tathum',
    version='0.4.1',
    description='TAT-HUM: Trajectory Analysis Toolkit for Human Movement',
    url='https://github.com/xywang01/TAT-HUM',
    download_url='https://github.com/xywang01/TAT-HUM/archive/refs/tags/0.4.tar.gz',
    author='X. Michael Wang, University of Toronto, Faculty of Kinesiology and Physical Education, AA Lab',
    author_email='michaelwxy.wang@utoronto.ca',
    license='MIT',
    packages=['tathum', ],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'typing',
        'numpy',
        'pandas',
        'scipy',
        'scikit-spatial',
        'vg',
        'pytransform3d',
        'matplotlib',
        'seaborn',
        'jupyter',
    ]
)
