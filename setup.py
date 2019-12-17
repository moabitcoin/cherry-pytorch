import setuptools
from setuptools import setup

base_url = 'https://github.com/moabitcoin/'

setup(name='cherry-rl',
      version_format='dev{commits}',
      setup_requires=['very-good-setuptools-git-version'],
      description='RL muse in Pytorch',
      url=base_url + 'cherry-pytorch',
      author='Harsimrat Sandhawalia',
      author_email='hs.sandhawalia@gmail.com',
      license='Moabitcoin Proprietry',
      packages=setuptools.find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
