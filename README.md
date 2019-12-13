# Deploy Transformers
Easily deploy Hugging Face's Transformers on a website


### Requirements:

* `pytorch`
* `transformers`

### Usage:

```python
from transformersDeploy.deploy import Deploy, list_model
# If you want to know model types and names: list_model()


model = Deploy(model_type="gpt2", model_name="distilgpt2", verbose=True)
model.generate(prompt="") # If no prompt, input will be ask until exit
```


### Notes:

* Essayez d'obtenir des prédictions de textes générés à partir d'un modèle
    préentraîné
* Deployez le modèle en local avec FastAPI (ou Flask si trop compliqué)
* Faire la même chose pour différentes tâches (classification, analysis...)
* Rendre automatique le déploiement de tout ça avec un package pip
* Faire un générateur de page HTML avec Python et f-string en remplacant les titres, messages, couleurs, etc...
* Intégrez spinners Halo
