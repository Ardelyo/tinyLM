# Custom configuration
from tinylm import TinyConfig, TinyLM, TinyTokenizer, TinyTrainer
import torch

# Customize config
config = TinyConfig()
config.hidden_size = 256  # Larger model
config.num_layers = 4
config.learning_rate = 1e-4

# Create components
tokenizer = TinyTokenizer(vocab_size=8000)
model = TinyLM(config)

# Train with custom settings
trainer = TinyTrainer(model, config, tokenizer)
trainer.train(num_samples=10000)

# Use the model
model.eval()
input_ids = torch.tensor([tokenizer.tokenize("Hello")])
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0].tolist()))
