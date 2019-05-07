from setuptools import setup

REQUIRED=[
      'scikit-learn>=0.20.2',
      'tensorflow>=1.8.0',
      'keras>=2.1.6',
      'pillow>=5.1.0',
      'matplotlib>=1.5.3',
      'pydot>=1.3.0',
      'pyclickhouse>=0.6.4',
      'lru-dict>=1.1.6',
      'pytest>=4.0.1',
      'dill>=0.2.9',
      'pytz>=2018.9',
      'tqdm>=4.31.1',
      'ujson>=1.35',
      'scandir>=1.10.0',
      'pymongo>=3.6.1',
      'numpy>=1.15.0'
]

setup(name='iwlearn',
      version='0.0.1',
      description='Immowelt Machine Learning Framework',
      url='https://github.com/Immowelt/iwlearn',
      download_url = 'https://github.com/Immowelt/iwlearn/archive/0.0.1.tar.gz',
      keywords = ['Scikit-Learn', 'Tensorflow', 'Keras', 'Machine Learning'],
      classifiers=[],
      author='Immowelt AG',
      author_email='info@immowelt.de',
      license='Apache2',
      packages=['iwlearn'],
      install_requires=REQUIRED,
      use_2to3=True,
      test_suite='tests',
      zip_safe=False)

