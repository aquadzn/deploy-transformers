from deploy_transformers import ListModels, Model

# ListModels() to show available models

model = Model("gpt2", "distilgpt2", seed=42, verbose=False)
model.generate(length=20, prompt="The quick brown fox jumps over the lazy dog")
# If no prompt, input will be ask until exit
