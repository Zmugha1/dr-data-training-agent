from setuptools import setup, find_packages

setup(
    name="dr_data_agents",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0.1",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
)
