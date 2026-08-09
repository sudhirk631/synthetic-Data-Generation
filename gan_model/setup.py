# this is redundant file. can be deleted.

from setuptools import find_packages, setup

setup(

    name='my_gan_package',

    version='0.1',

    packages=find_packages(),

    install_requires=[

        'db-dtypes',

        'tensorflow==2.19.0',

        'pandas',

        'numpy',

        'scikit-learn',

        'psutil',

        'scipy',

        'google-cloud-bigquery',

        'gcsfs'

    ],

    include_package_data=True,

    description='GAN for synthetic data',

)
