from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-classification-challenge",
    version="1.0.0",
    author="Medical AI Challenge Team",
    author_email="team@medical-ai-challenge.com",
    description="Sistema de clasificación automática de literatura médica usando Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/medical-classification-challenge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
        "api": [
            "gunicorn>=20.0",
            "flask-cors>=3.0",
            "flask-limiter>=1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-classifier-train=main:main",
            "medical-classifier-api=scripts.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/tu-usuario/medical-classification-challenge/issues",
        "Source": "https://github.com/tu-usuario/medical-classification-challenge",
        "Documentation": "https://github.com/tu-usuario/medical-classification-challenge/wiki",
    },
)
