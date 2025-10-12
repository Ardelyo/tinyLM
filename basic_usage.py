# Train and use in 5 lines
from tinylm import TinyLMManager

manager = TinyLMManager()
manager.create_model()
manager.train_model(1000)
inference = manager.get_inference_engine()
print(inference.generate_text("Hello world"))
