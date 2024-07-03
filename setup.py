import setuptools

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

__version__ = '0.0.0'

NAME="SETI Signals Classifier"
REPO_NAME="End-to-End-SETI-Classification-using-CNN-MLFlow-DVC"
AUTHOR_USER_NAME="bhuvaneshprasad"
SRC_REPO="cnnClassifier"
AUTHOR_EMAIL="pbhuvanesh3@gmail.com"

setuptools.setup(
    name=NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Extraterrestrial Intelligence Signals Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
