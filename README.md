# 🚀 Deploy Transformers
> Easily deploy HuggingFace's 🤗 Transformers on a website


One to two paragraph statement about your product and what it does.


## Installation

**Pytorch** and **Transformers** are obviously needed.

```bash
pip install transformersDeploy
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

You can either **clone** this repository to have original files or **create yourself** the structure or use the function `website.create_structure()`.

This will automatically create *templates/*, *static/* and all the files that are in it if none of them exist.


## Usage

![Generation snippets](https://svgshare.com/i/Gne.svg)![Deployment snippets](https://svgshare.com/i/GnN.svg)

**You can change homepage filename, templates/ and static/ names in `website.deploy()` but it's better to keep them as default.**


## Notes

* Essayez d'obtenir des prédictions de textes générés à partir d'un modèle
    préentraîné
* Deployez le modèle en local avec Flask (voir pour rajouter Django, FastAPI si Flask fini)
* Faire la même chose pour différentes tâches (classification, analysis...)
* Rendre automatique le déploiement de tout ça avec un package pip
* Faire un générateur de page HTML avec dans deploy.py et f-string en remplacant les titres, messages, couleurs, etc...
* Intégrez spinners Halo