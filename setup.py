from setuptools import setup, find_packages

setup(
    name="grapheaglevision",
    version="0.1.0",
    description="GraphEagleVision: Structural Cohesiveness Dynamics for Temporal Link Prediction",
    packages=find_packages(include=["gev", "gev.*", "integration", "integration.*"]),
    python_requires=">=3.9",
)
