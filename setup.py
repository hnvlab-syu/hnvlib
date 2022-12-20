from setuptools import setup, find_packages

from hnvlib import __version__


def get_install_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith('#')]
    return reqs


def get_long_description():
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()
    return long_description


if __name__ == '__main__':
    setup(
        name='hnvlib',
        version=__version__,
        description='Standard code library for HnV Lab.',
        author='Summer Lee',
        maintainer='Summer Lee',
        maintainer_email='leeyeoreum01@gmail.com',
        url='https://github.com/HnV-Lab/hnvlib',
        # license='BSD',
        packages=find_packages(),
        # Sympy 1.4 is needed for printing tests to pass, but 1.3 will still work
        install_requires=get_install_requirements(),
        python_requires='<3.9',
        long_description=get_long_description(),
        classifiers=[
            # 'Development Status :: 4 - Beta',
            # 'Environment :: Console',
            # 'Intended Audience :: Science/Research',
            # 'License :: OSI Approved :: BSD License',
            'Natural Language :: Korean',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            # 'Programming Language :: Python :: 3.9',
            # 'Topic :: Scientific/Engineering :: Mathematics',
            # 'Topic :: Scientific/Engineering :: Physics',
        ],
        project_urls={
            'Documentation': 'https:w//hnvlib.readthedocs.io',
            # 'Bug Tracker': 'https://github.com/HnV-Lab/hnvlib/issues',
            'Source Code': 'https://github.com/HnV-Lab/hnvlib',
        },
    )
