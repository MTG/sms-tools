from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sms-tools",
    version="0.9",
    author="Music Technology Group",
    author_email="mtg-info@upf.edu",
    description="tools for sound analysis/synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MTG/sms-tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
