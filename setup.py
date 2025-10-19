
from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
with open("VERSION", "r", encoding="utf-8") as fh:
    version = fh.read().strip()

setup(
    name="chest-xray-detection",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Chest X-Ray Pathology Detection using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/chest_xray_detection",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/chest_xray_detection/issues",
        "Documentation": "https://github.com/your-username/chest_xray_detection/docs",
        "Source Code": "https://github.com/your-username/chest_xray_detection",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'cxr-analyze=scripts.01_analyze_data:main',
            'cxr-train=scripts.02_train_model:main',
            'cxr-evaluate=scripts.03_evaluate_model:main',
            'cxr-predict=scripts.04_predict:main',
        ],
    },
    include_package_data=True,
    package_data={
        "chest_xray_detection": [
            "configs/*.yaml",
            "examples/*.jpg",
            "examples/*.png",
        ],
    },
    keywords=[
        "medical-ai",
        "computer-vision", 
        "chest-xray",
        "yolo",
        "object-detection",
        "healthcare",
    ],
    license="MIT",
    platforms=["any"],
)
