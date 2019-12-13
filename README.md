# Deploy Transformers
Easily deploy Hugging Face's Transformers on a website


### Requirements:

* `pytorch`
* `transformers`
* ``FastAPI` or `Flask`

### Usage:

```python
from transformersDeploy.deploy import list_models, Model, Deploy
# If you want to know model types and names: list_model()


model = Model(model_type="gpt2", model_name="distilgpt2", verbose=True)
model.generate(prompt="") # If no prompt, input will be ask until exit
```


### Notes:

* Essayez d'obtenir des prédictions de textes générés à partir d'un modèle
    préentraîné
* Deployez le modèle en local avec Flask (voir pour rajouter Django, FastAPI si Flask fini)
* Faire la même chose pour différentes tâches (classification, analysis...)
* Rendre automatique le déploiement de tout ça avec un package pip
* Faire un générateur de page HTML avec dans deploy.py et f-string en remplacant les titres, messages, couleurs, etc...
* Intégrez spinners Halo