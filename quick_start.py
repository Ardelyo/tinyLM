"""
quick_start.py - Get TinyLM running in 30 seconds!
"""

from tinylm import TinyLMManager

def quick_start():
    print("🚀 TinyLM Quick Start")
    print("=" * 40)
    
    # Initialize manager
    manager = TinyLMManager()
    
    # Create model
    print("Creating model...")
    manager.create_model()
    
    # Train with minimal data
    print("Training (this will take ~1 minute)...")
    manager.train_model(1000)  # Small dataset for quick start
    
    # Get inference engine
    inference = manager.get_inference_engine()
    
    # Test generation
    print("\n✨ Model Ready! Testing generation:")
    print("-" * 40)
    
    test_prompts = [
        "Hello, how are",
        "The weather is",
        "I like to",
        "Technology is"
    ]
    
    for prompt in test_prompts:
        generated = inference.generate_text(prompt, max_length=20)
        print(f"'{prompt}' → '{generated}'")
    
    # Save model
    print("\n💾 Saving model...")
    manager.save_model('tinylm_quickstart')
    
    print("\n✅ Quick start completed!")
    print("You can now use: python tinylm.py chat")

if __name__ == "__main__":
    quick_start()
