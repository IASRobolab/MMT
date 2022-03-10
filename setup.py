from setuptools import setup, find_packages


setup(name='MMT',
      version='1.1.0',
      description='Pytorch Library of Mutual Mean-Teaching for Unsupervised Domain Adaptation on Person Re-identification',
      author='Yixiao Ge',
      author_email='yxge@link.cuhk.edu.hk',
      url='https://github.com/yxgeee/MMT.git',
      install_requires=[
          'numpy', 
          'six', 'h5py', 'Pillow', 'scipy', 'torch>=1.4.0', 'torchvision>=0.5.0',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Deep Learning',
      ])
