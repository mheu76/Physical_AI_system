"""
Physical AI System 설치 스크립트
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt 파일 읽기
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="physical-ai-system",
    version="1.0.0",
    author="Physical AI Team",
    author_email="team@physical-ai.com",
    description="발달적 학습과 체화된 지능을 구현하는 Physical AI 시스템",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/physical-ai-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "simulation": [
            "pybullet>=3.2.5",
            "mujoco>=2.3.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
            "dash>=2.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "physical-ai=main:main",
            "physical-ai-test=tests.test_integration:run_integration_tests",
            "physical-ai-example=examples.basic_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "physical_ai_system": [
            "configs/*.yaml",
            "models/*.urdf",
            "models/*.xml",
        ],
    },
    keywords=[
        "artificial intelligence",
        "robotics", 
        "developmental learning",
        "embodied AI",
        "physical AI",
        "machine learning",
        "reinforcement learning",
        "developmental robotics"
    ],
)
