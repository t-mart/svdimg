from setuptools import setup, find_packages

setup(
        name='svdimg',
        version='0.1',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "click==4.0",
            "numpy==1.9.2",
            "Pillow==2.9.0",
            "scipy==0.15.1",
        ],
        entry_points={
            'console_scripts': [
                'svdimg=svdimg.svdimg:main',
            ],
        }
)
