from setuptools import setup


with open('requirements.txt') as f:
    requirements_list = [line.strip() for line in f.readlines()]


setup(
    install_requires=requirements_list,
    extras_require={
         'develop': ['flake8>=4.0.1', 'pytest>=6.2.5'],
    },
)
