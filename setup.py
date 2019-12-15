import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformers-deploy", # Replace with your own username
    version="0.1",
    author="William Jacques",
    author_email="williamjcqs8@gmail.com",
    description="Easily deploy HuggingFace Transformers on a website",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aquadzn/deploy-transformers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'transformers',
        'starlette',
        'uvicorn',
        'jinja2',
        'aiofiles'
    ]
)