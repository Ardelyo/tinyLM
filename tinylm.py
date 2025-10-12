"""
TinyLM - A Complete Tiny Language Model
No external data needed, fully functional, <10MB model size
Author: TinyLM Project
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import random
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
import pickle
import gzip

# ================== CONFIGURATION ==================
class TinyConfig:
    def __init__(self):
        self.vocab_size = 5000
        self.hidden_size = 128
        self.num_layers = 3
        self.num_heads = 4
        self.max_length = 128
        self.dropout = 0.1
        self.learning_rate = 3e-4
        self.batch_size = 16
        self.epochs = 10
        self.temperature = 0.8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== TOKENIZER ==================
class TinyTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        
        # Build basic vocabulary
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
        
    def _build_vocab(self):
        """Build a basic vocabulary with common words and subwords"""
        # Special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Common English words (top 1000)
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
            'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
            'may', 'might', 'shall', 'should', 'must', 'can', 'could', 'need', 'able', 'going',
            'hello', 'hi', 'yes', 'no', 'please', 'thank', 'sorry', 'okay', 'oh', 'well',
            'very', 'really', 'quite', 'much', 'more', 'less', 'too', 'enough', 'such', 'rather'
        ]
        
        # Add common words
        idx = len(self.special_tokens)
        for word in common_words[:self.vocab_size - 1000]:  # Reserve space for subwords
            if word not in self.vocab:
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word
                idx += 1
        
        # Add character-level tokens and common subwords
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"- \n'
        for char in chars:
            if char not in self.vocab and idx < self.vocab_size - 500:
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
                idx += 1
        
        # Common prefixes and suffixes
        subwords = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'able', 'ful', 'less', 'ness',
                   'ment', 'ship', 'ward', 'wise', 'un', 're', 'pre', 'post', 'anti', 'non']
        
        for subword in subwords:
            if idx < self.vocab_size:
                self.vocab[subword] = idx
                self.reverse_vocab[idx] = subword
                idx += 1
        
        # Fill remaining with numbered tokens
        while idx < self.vocab_size:
            token = f'[TOKEN_{idx}]'
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            idx += 1
    
    def tokenize(self, text: str) -> List[int]:
        """Simple word and character-based tokenization"""
        text = text.lower().strip()
        tokens = []
        
        # Add BOS token
        tokens.append(self.vocab[self.bos_token])
        
        # Split by spaces first
        words = text.split()
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback for unknown words
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab[self.unk_token])
        
        # Add EOS token
        tokens.append(self.vocab[self.eos_token])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if token not in self.special_tokens and not token.startswith('[TOKEN_'):
                    words.append(token)
        
        # Smart joining
        text = ''
        for i, word in enumerate(words):
            if len(word) == 1 and word in '.,!?;:':
                text += word
            elif i == 0:
                text += word
            else:
                text += ' ' + word
        
        return text.strip()

# ================== MODEL ARCHITECTURE ==================
class TinyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(scores.device) * -1e9
        scores = scores + causal_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class TinyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.fc2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TinyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TinyAttention(config)
        self.mlp = TinyMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TinyTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50, top_p=0.95):
        """Generate text using the model"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for the last token
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[torch.arange(next_token_logits.size(0))[:, None], indices_to_remove] = -float('inf')
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if EOS token is generated
                if next_token.item() == 3:  # EOS token ID
                    break
        
        return generated

# ================== DATA GENERATION ==================
class TinyDataGenerator:
    """Generate synthetic training data without external dependencies"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.templates = self._create_templates()
        
    def _create_templates(self):
        """Create diverse text templates for training"""
        return {
            'greetings': [
                "hello how are you",
                "hi there nice to meet you",
                "good morning how is your day",
                "hey what's up",
                "greetings and welcome"
            ],
            'questions': [
                "what is your name",
                "where are you from",
                "how old are you",
                "what do you like",
                "when did this happen"
            ],
            'statements': [
                "the weather is nice today",
                "i like to read books",
                "technology is advancing rapidly",
                "learning is important",
                "the world is beautiful"
            ],
            'responses': [
                "yes i agree with you",
                "no i don't think so",
                "maybe we should consider",
                "that sounds interesting",
                "i understand your point"
            ],
            'descriptions': [
                "the cat is sleeping on the mat",
                "the sun is shining bright",
                "the tree has green leaves",
                "the river flows gently",
                "the mountain is very tall"
            ],
            'conversations': [
                "hello there how can i help you today",
                "i need some assistance with my work",
                "sure i would be happy to help",
                "thank you very much for your time",
                "you are welcome have a great day"
            ],
            'instructions': [
                "please follow these steps carefully",
                "first you need to prepare",
                "then you should proceed with",
                "finally make sure to check",
                "remember to save your work"
            ],
            'stories': [
                "once upon a time there was",
                "in a land far away",
                "the hero went on a journey",
                "they faced many challenges",
                "and lived happily ever after"
            ]
        }
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data"""
        data = []
        
        for _ in range(num_samples):
            # Randomly select category and template
            category = random.choice(list(self.templates.keys()))
            template = random.choice(self.templates[category])
            
            # Create variations
            text = self._create_variation(template)
            
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            
            if len(tokens) > 2:  # Ensure minimum length
                data.append(tokens)
        
        return data
    
    def _create_variation(self, template):
        """Create variations of templates"""
        variations = [
            template,
            template.upper(),
            template.lower(),
            template.capitalize(),
            ' '.join(template.split()[::-1]),  # Reverse word order
            template.replace(' ', ' really '),  # Add emphasis
            template + ' and more',  # Add continuation
            'well ' + template,  # Add prefix
            template + ' right',  # Add suffix
        ]
        
        return random.choice(variations)
    
    def create_batch(self, data, batch_size, max_length):
        """Create a batch of padded sequences"""
        batch = random.sample(data, min(batch_size, len(data)))
        
        # Pad sequences
        padded_batch = []
        for seq in batch:
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [0] * (max_length - len(seq))  # Pad with zeros
            padded_batch.append(seq)
        
        return torch.tensor(padded_batch, dtype=torch.long)

# ================== TRAINING ==================
class TinyTrainer:
    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.data_generator = TinyDataGenerator(tokenizer)
        
    def train(self, num_samples=5000):
        """Train the model with synthetic data"""
        print("🚀 Starting TinyLM Training...")
        print(f"Device: {self.config.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        # Generate training data
        print("📝 Generating synthetic training data...")
        training_data = self.data_generator.generate_training_data(num_samples)
        
        self.model.train()
        self.model.to(self.config.device)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Create batches
            for _ in range(num_samples // self.config.batch_size):
                # Get batch
                batch = self.data_generator.create_batch(
                    training_data, 
                    self.config.batch_size, 
                    self.config.max_length
                ).to(self.config.device)
                
                # Shift for language modeling
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=0  # Ignore padding
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch+1}/{self.config.epochs} | Loss: {avg_loss:.4f}")
        
        print("✅ Training completed!")
        return self.model

# ================== INFERENCE ENGINE ==================
class TinyInference:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        
    def generate_text(self, prompt, max_length=50, temperature=0.8):
        """Generate text from a prompt"""
        # Tokenize prompt
        input_ids = self.tokenizer.tokenize(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=temperature
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        return generated_text
    
    def chat(self, message):
        """Simple chat interface"""
        response = self.generate_text(message, max_length=30)
        return response
    
    def complete(self, text):
        """Text completion"""
        return self.generate_text(text, max_length=20, temperature=0.7)

# ================== MODEL MANAGER ==================
class TinyLMManager:
    def __init__(self):
        self.config = TinyConfig()
        self.tokenizer = TinyTokenizer(vocab_size=self.config.vocab_size)
        self.model = None
        
    def create_model(self):
        """Create a new model"""
        self.model = TinyLM(self.config)
        print(f"✨ Created TinyLM with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
        return self.model
    
    def train_model(self, num_samples=5000):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        trainer = TinyTrainer(self.model, self.config, self.tokenizer)
        self.model = trainer.train(num_samples)
        return self.model
    
    def save_model(self, path='tinylm_model.pth'):
        """Save model to disk"""
        if self.model is None:
            print("❌ No model to save!")
            return
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'tokenizer_vocab': self.tokenizer.vocab,
            'tokenizer_reverse': self.tokenizer.reverse_vocab
        }
        
        # Compress and save
        with gzip.open(path + '.gz', 'wb') as f:
            pickle.dump(save_dict, f)
        
        size_mb = os.path.getsize(path + '.gz') / (1024 * 1024)
        print(f"💾 Model saved to {path}.gz ({size_mb:.2f}MB)")
    
    def load_model(self, path='tinylm_model.pth'):
        """Load model from disk"""
        if not os.path.exists(path + '.gz'):
            print(f"❌ Model file {path}.gz not found!")
            return None
        
        with gzip.open(path + '.gz', 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore config
        self.config = TinyConfig()
        for key, value in save_dict['config'].items():
            setattr(self.config, key, value)
        
        # Restore tokenizer
        self.tokenizer = TinyTokenizer(vocab_size=self.config.vocab_size)
        self.tokenizer.vocab = save_dict['tokenizer_vocab']
        self.tokenizer.reverse_vocab = save_dict['tokenizer_reverse']
        
        # Restore model
        self.model = TinyLM(self.config)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {path}.gz")
        return self.model
    
    def get_inference_engine(self):
        """Get inference engine"""
        if self.model is None:
            print("❌ No model loaded!")
            return None
        
        return TinyInference(self.model, self.tokenizer, self.config)

# ================== INTERACTIVE CLI ==================
class TinyLMCLI:
    def __init__(self):
        self.manager = TinyLMManager()
        self.inference = None
        
    def run(self):
        """Run interactive CLI"""
        print('''
╔════════════════════════════════════════╗
║          🤏 TinyLM v1.0.0              ║
║   Tiny Language Model - Ready to Use!  ║
╚════════════════════════════════════════╝
        ''')
        
        while True:
            print("\n" + "="*40)
            print("Commands:")
            print("1. Create new model")
            print("2. Train model")
            print("3. Generate text")
            print("4. Chat mode")
            print("5. Save model")
            print("6. Load model")
            print("7. Model info")
            print("8. Quick demo")
            print("9. Exit")
            print("="*40)
            
            choice = input("\nEnter choice (1-9): ").strip()
            
            if choice == '1':
                self.create_model()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.generate_text()
            elif choice == '4':
                self.chat_mode()
            elif choice == '5':
                self.save_model()
            elif choice == '6':
                self.load_model()
            elif choice == '7':
                self.model_info()
            elif choice == '8':
                self.quick_demo()
            elif choice == '9':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice!")
    
    def create_model(self):
        """Create new model"""
        self.manager.create_model()
        print("✅ Model created successfully!")
    
    def train_model(self):
        """Train model"""
        if self.manager.model is None:
            print("📝 Creating new model first...")
            self.manager.create_model()
        
        samples = input("Number of training samples (default 5000): ").strip()
        samples = int(samples) if samples else 5000
        
        print("🏋️ Training model... (this may take a few minutes)")
        self.manager.train_model(samples)
        self.inference = self.manager.get_inference_engine()
        print("✅ Training completed!")
    
    def generate_text(self):
        """Generate text"""
        if self.inference is None:
            print("❌ Please train or load a model first!")
            return
        
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            prompt = "Hello"
        
        print("\n🤖 Generating...")
        result = self.inference.generate_text(prompt)
        print(f"\n📝 Generated: {result}")
    
    def chat_mode(self):
        """Interactive chat"""
        if self.inference is None:
            print("❌ Please train or load a model first!")
            return
        
        print("\n💬 Chat Mode (type 'exit' to quit)")
        print("-" * 40)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            response = self.inference.chat(user_input)
            print(f"TinyLM: {response}")
    
    def save_model(self):
        """Save model"""
        if self.manager.model is None:
            print("❌ No model to save!")
            return
        
        filename = input("Enter filename (default: tinylm_model): ").strip()
        filename = filename if filename else "tinylm_model"
        self.manager.save_model(filename)
    
    def load_model(self):
        """Load model"""
        filename = input("Enter filename (default: tinylm_model): ").strip()
        filename = filename if filename else "tinylm_model"
        
        if self.manager.load_model(filename):
            self.inference = self.manager.get_inference_engine()
            print("✅ Model loaded and ready!")
    
    def model_info(self):
        """Display model information"""
        if self.manager.model is None:
            print("❌ No model loaded!")
            return
        
        total_params = sum(p.numel() for p in self.manager.model.parameters())
        trainable_params = sum(p.numel() for p in self.manager.model.parameters() if p.requires_grad)
        
        print("\n" + "="*40)
        print("📊 Model Information")
        print("="*40)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f}MB (float32)")
        print(f"Vocabulary size: {self.manager.config.vocab_size}")
        print(f"Hidden size: {self.manager.config.hidden_size}")
        print(f"Number of layers: {self.manager.config.num_layers}")
        print(f"Number of attention heads: {self.manager.config.num_heads}")
        print(f"Max sequence length: {self.manager.config.max_length}")
    
    def quick_demo(self):
        """Quick demonstration"""
        print("\n🎯 Running Quick Demo...")
        print("-" * 40)
        
        # Create and train small model
        print("1️⃣ Creating model...")
        self.manager.create_model()
        
        print("2️⃣ Training with 1000 samples...")
        self.manager.train_model(1000)
        self.inference = self.manager.get_inference_engine()
        
        print("3️⃣ Generating text samples...")
        prompts = ["hello", "the weather", "i like", "what is"]
        
        for prompt in prompts:
            result = self.inference.generate_text(prompt, max_length=20)
            print(f"   '{prompt}' → '{result}'")
        
        print("\n✅ Demo completed!")

# ================== MAIN EXECUTION ==================
def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'train':
            # Quick training
            manager = TinyLMManager()
            manager.create_model()
            manager.train_model(5000)
            manager.save_model('tinylm_trained')
            print("✅ Model trained and saved!")
            
        elif command == 'generate':
            # Quick generation
            manager = TinyLMManager()
            if manager.load_model('tinylm_trained'):
                inference = manager.get_inference_engine()
                prompt = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello"
                result = inference.generate_text(prompt)
                print(f"Generated: {result}")
            else:
                print("❌ Please train a model first: python tinylm.py train")
                
        elif command == 'chat':
            # Quick chat
            manager = TinyLMManager()
            if manager.load_model('tinylm_trained'):
                inference = manager.get_inference_engine()
                print("💬 Chat Mode (type 'exit' to quit)")
                while True:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    response = inference.chat(user_input)
                    print(f"TinyLM: {response}")
            else:
                print("❌ Please train a model first: python tinylm.py train")
                
        elif command == 'help':
            print('''
TinyLM - Tiny Language Model

Usage:
    python tinylm.py              # Interactive mode
    python tinylm.py train        # Train new model
    python tinylm.py generate     # Generate text
    python tinylm.py chat         # Chat mode
    python tinylm.py help         # Show this help
            ''')
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python tinylm.py help' for usage information")
    else:
        # Interactive mode
        cli = TinyLMCLI()
        cli.run()

if __name__ == "__main__":
    main()
