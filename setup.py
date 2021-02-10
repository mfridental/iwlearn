from setuptools import setup

REQUIRED=[
      'scikit-learn>=0.23.2',
      'tensorflow>=2.3.1',
      'keras>=2.4.3',
      'pillow>=8.0.0',
      'matplotlib>=3.3.2',
      'pydot>=1.4.1',
      'pyclickhouse>=0.6.4',
      'lru-dict>=1.1.6',
      'pytest>=6.1.1',
      'dill>=0.3.2',
      'pytz>=2020.1',
      'tqdm>=4.50.2',
      'ujson>=4.0.1',
      'scandir>=1.10.0',
      'pymongo>=3.11.0',
      'numpy>=1.19.2',
      'lru-dict>=1.1.6'
]

setup(name='iwlearn3',
      version='0.1.1',
      description='Immowelt Machine Learning Framework (Python 3)',
      url='https://github.com/mfridental/iwlearn',
      download_url = 'https://github.com/mfridental/iwlearn/archive/0.1.1.tar.gz',
      keywords = ['Scikit-Learn', 'Tensorflow', 'Keras', 'Machine Learning'],
      classifiers=[
            "Programming Language :: Python :: 3"
      ],
      author='Maxim Fridental (Maintainer)',
      author_email='maxim@fridental.de',
      license='Apache2',
      packages=['iwlearn'],
      install_requires=REQUIRED,
      test_suite='tests',
      zip_safe=False)

