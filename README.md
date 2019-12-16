# 🚀 Deploy Transformers 🤗
> Deploy a SOTA model for text-generation in just three lines of code 💻
> 
![image](https://svgshare.com/i/GnD.svg)


## Installation 

[**Pytorch**](https://pytorch.org/get-started/locally/#start-locally) and [**Transformers**](https://github.com/huggingface/transformers/#installation) are obviously needed.

```bash
pip install deploy-transformers
```

**For deployment, file structure needs to be like this:**
```bash
├── static
│   ├── script.js
│   ├── style.css
├── templates
│   ├── 404.html
│   ├── index.html
|
└── your_file.py
```

You can either **clone** this repository to have original files or use the function `website.create_structure()` or **create yourself** the structure.

`website.create_structure()` will automatically create *templates/*, *static/* and all the files that are in it (.html, .js, .css).


## Usage

Check the *[examples/](https://github.com/aquadzn/deploy-transformers/tree/master/examples)* folder.

```python
# Deployment
from deploy_transformers import Website

website = Website(model_type="gpt2", model_name="distilgpt2")
# website.create_folder(homepage_file="index.html", template_folder='templates', static_folder='static')
website.deploy()
```

**You can change homepage filename, templates/ and static/ names in `website.deploy()` but it's better to keep them as default.**

```python
# Only text generation
from deploy_transformers import ListModels, Model

# ListModels() to show available models
model = Model("gpt2", "distilgpt2", seed=42, verbose=False)
model.generate(length=20, prompt="The quick brown fox jumps over the lazy dog")
# If no prompt, input will be ask until exit
```

There is also a Dockerfile.

## Thanks

* [Transformers](https://github.com/huggingface/transformers) package by HuggingFace
* [gpt-2-cloudrun](https://github.com/minimaxir/gpt-2-cloud-run) by minimaxir

## Notes

* Do the same but for other tasks like sentiment analysis, or Q&A.
* Add Flask option?
