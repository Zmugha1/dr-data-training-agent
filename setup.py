"""Setup for Human-in-the-Loop Decision Intelligence (70:20:10)."""

from setuptools import setup, find_packages

setup(
    name="dr_data_agents",
    version="0.1.0",
    description="Human-in-the-Loop Decision Intelligence for Training Transfer Optimization",
    long_description=open("README.md", encoding="utf-8").read() if __import__("pathlib").Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "data*"]),
    python_requires=">=3.10",
    install_requires=[
        "streamlit>=1.28.0",
        "pydantic>=2.0.0",
        "plotly>=5.18.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "hitl-streamlit=ui.streamlit_app:main",
        ],
    },
)
