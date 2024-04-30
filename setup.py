from setuptools import setup

setup(
    name='LoG',     
    version='0.0',   #
    description='Level of Gaussians',
    author='Qing Shuai', 
    author_email='s_q@zju.edu.cn',
    # test_suite='setup.test_all',
    packages=[
        'LoG',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    install_requires=[],
    data_files = []
)