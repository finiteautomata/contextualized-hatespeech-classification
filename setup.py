from setuptools import setup, find_packages

setup(
    name='hatespeech-classification',
    version='0.0.1',
    author="Juan Manuel Pérez",
    author_email="jmperez@dc.uba.ar",
    description="A Transformer-based library for hate speech detection in Spanish",
    packages=["hatedetection"],
    test_suite="tests",
    install_requires=[
        "transformers==4.4.0",
        "datasets==1.4.1",
	    "torch",
        "emoji",
        "pandarallel",
    ]
)
