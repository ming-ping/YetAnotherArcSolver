from distutils.core import setup

setup(name='yarc',
      version='0.1.1',
      description='Solve The Abstraction and Reasoning Corpus for Artificial General Intelligence',
      author='yarc contributors',
      author_email='mingping765@gmailcom',
      license='mit',
      packages=['yarc'],
      url="https://github.com/ming-ping/YetAnotherArcSolver",
      install_requires=[
          'numpy',
          'scipy',
          'torch'
      ],
      package_data={'yarc': ['models/*.pt', 'models/*.gz']}
      )
