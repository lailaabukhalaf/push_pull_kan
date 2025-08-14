import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="push-pull-kan",
    version="0.1.0",
    author="Layla Abu Khalaf",
    author_email="your.email@example.com",
    description="Push-Pull Kolmogorov Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["Model"],  # <-- treat Model as the package
    package_dir={"Model": "Model"},
    include_package_data=True,
    python_requires='>=3.6',
)
