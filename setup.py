from setuptools import setup

setup(
    name='tathum',
    version='0.1',
    description='TAT-HUM: Trajectory Analysis Toolkit for Human Movement',
    url='https://github.com/xywang01/TAT-HUM',
    download_url='',
    author='X. Michael Wang, University of Toronto, Faculty of Kinesiology and Physical Education, AA Lab',
    author_email='michaelwxy.wang@utoronto.ca',
    license='MIT',
    packages=['tathum', ],
    zip_safe=False,
    install_requires=[
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
