import re

import setuptools

setuptools.setup(
    name="ouch",
    version=re.compile(r"__version__\s*=\s*['\"](.*)['\"]").findall(
        open("ouch/__init__.py", "r").read()
    )[0],
    description="Odd Utility Collection Hub",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thyeem/ouch",
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    license="MIT",
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="utilities functional",
    packages=setuptools.find_packages(),
    install_requires=["foc"],
    python_requires=">=3.6",
)
