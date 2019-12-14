# Deploy Transformers
Easily deploy Hugging Face's Transformers on a website


### Requirements
---

* `pytorch`
* `transformers`
* `uvicorn` `starlette` `jinja2` `aiofiles`


### Usage
---

**Text generation**
```python
from transformersDeploy.deploy import Model, ListModels, Website

# ListModels() shows available models

model = Model("gpt2", "distilgpt2", seed=42, verbose=False)
model.generate(length=20, prompt="The quick brown fox jumps over the lazy dog")
# If no prompt, input will be ask until exit
```

**Deploy app**
```python
from transformersDeploy.deploy import Model, ListModels, Website


website = Website(model_type="gpt2", model_name="distilgpt2")
website.deploy(homepage_file="index.html")
```


### Notes
---

* Essayez d'obtenir des prédictions de textes générés à partir d'un modèle
    préentraîné
* Deployez le modèle en local avec Flask (voir pour rajouter Django, FastAPI si Flask fini)
* Faire la même chose pour différentes tâches (classification, analysis...)
* Rendre automatique le déploiement de tout ça avec un package pip
* Faire un générateur de page HTML avec dans deploy.py et f-string en remplacant les titres, messages, couleurs, etc...
* Intégrez spinners Halo