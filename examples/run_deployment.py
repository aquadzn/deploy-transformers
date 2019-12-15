from deploy_transformers import Website

website = Website(model_type="gpt2", model_name="distilgpt2")
# website.create_folder(homepage_file="index.html", template_folder='templates', static_folder='static')
website.deploy()