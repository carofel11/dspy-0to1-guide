# Installation & Hello World

Get up and running with DSPy in minutes. This guide covers installation, configuration, and your first working example.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- (Optional) OpenAI API key or local LLM setup

### Installation Options

#### Option 1: Using this Repository
```bash
# Clone the repository
git clone <repository-url>
cd dspy-0to1-guide

# Set up development environment
make dev-setup

# Activate virtual environment
source .venv/bin/activate

# Run your first example
make quick-start
```

#### Option 2: Standalone Installation
```bash
# Basic installation
pip install dspy-ai

# With additional dependencies
pip install dspy-ai[all]

# Development installation
pip install dspy-ai transformers sentence-transformers faiss-cpu
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# Required for OpenAI models
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Local models
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Model Configuration

#### Cloud Models (Recommended for Getting Started)
```python
import dspy

# OpenAI GPT models
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
dspy.configure(lm=dspy.LM('openai/gpt-4'))

# Anthropic Claude
dspy.configure(lm=dspy.LM('anthropic/claude-3-sonnet'))

# Cohere
dspy.configure(lm=dspy.LM('cohere/command-r-plus'))
```

#### Local Models with Ollama
```bash
# Install Ollama (macOS)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run a model
ollama pull llama3
ollama serve
```

```python
import dspy

# Configure DSPy to use local Ollama
dspy.configure(lm=dspy.LM('ollama_chat/llama3'))
```

## 👋 Hello World Example

Let's create your first DSPy program - a simple question-answering system:

### Basic Example

```python
import dspy

# Configure the language model
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

class SimpleQA(dspy.Module):
    """Simple question-answering module using ChainOfThought."""
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str):
        return self.generate_answer(question=question)

# Create and use the module
qa = SimpleQA()
response = qa("What is the capital of France?")

print("Question:", "What is the capital of France?")
print("Answer:", response.answer)
if hasattr(response, 'rationale'):
    print("Reasoning:", response.rationale)
```

**Expected Output:**
```
Question: What is the capital of France?
Answer: Paris
Reasoning: The capital of France is Paris, which is the largest city in France and serves as the country's political, economic, and cultural center.
```

### Math QA Example

Here's a more sophisticated example for mathematical reasoning:

```python
import dspy

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

class MathQA(dspy.Module):
    """Mathematical reasoning with step-by-step solution."""
    
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("question -> answer: float")
    
    def forward(self, question: str):
        result = self.solve(question=question)
        return result

# Test with a math problem
math_qa = MathQA()
problem = "What is 15% of 240, plus 30?"

result = math_qa(problem)
print(f"Problem: {problem}")
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.rationale}")
```

**Expected Output:**
```
Problem: What is 15% of 240, plus 30?
Answer: 66.0
Reasoning: First, I need to calculate 15% of 240. 15% = 0.15, so 0.15 × 240 = 36. Then I add 30: 36 + 30 = 66.
```

## 🔧 Testing Your Installation

### Quick Verification
Save this as `test_installation.py`:

```python
import dspy
import os

def test_basic_functionality():
    """Test basic DSPy functionality."""
    try:
        # Configure with a simple model
        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
        
        # Create a simple module
        predictor = dspy.Predict("input -> output")
        result = predictor(input="Hello DSPy!")
        
        print("✅ DSPy is working correctly!")
        print(f"Test result: {result.output}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_environment():
    """Check environment configuration."""
    issues = []
    
    # Check API keys
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OPENAI_API_KEY not found in environment")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 9):
        issues.append(f"Python {sys.version} is too old. Requires 3.9+")
    
    if issues:
        print("⚠️  Environment issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Environment looks good!")
    
    return len(issues) == 0

if __name__ == "__main__":
    print("Testing DSPy installation...")
    env_ok = test_environment()
    
    if env_ok:
        func_ok = test_basic_functionality()
        if func_ok:
            print("\n🎉 DSPy is ready to use!")
        else:
            print("\n🔧 Check your configuration and try again.")
    else:
        print("\n🔧 Fix environment issues and try again.")
```

Run the test:
```bash
python test_installation.py
```

## 🎯 Next Steps

Once you have DSPy running, here are recommended next steps:

### 1. Run the Repository Examples
```bash
# Basic examples
make run-basic

# Advanced examples  
make run-advanced

# All examples
make run-all-examples
```

### 2. Explore Different Models
Try switching between different models to see how DSPy's model-agnostic design works:

```python
# Try different models with the same code
models_to_test = [
    'openai/gpt-4o-mini',
    'openai/gpt-4',
    'anthropic/claude-3-sonnet',
    'ollama_chat/llama3'  # if you have Ollama running
]

for model in models_to_test:
    print(f"\nTesting with {model}:")
    dspy.configure(lm=dspy.LM(model))
    qa = SimpleQA()
    result = qa("What is machine learning?")
    print(f"Answer: {result.answer[:100]}...")
```

### 3. Experiment with Signatures
Try different signature styles:

```python
# String-based signature
simple_sig = dspy.Predict("question -> answer")

# Detailed class-based signature
class DetailedQA(dspy.Signature):
    """Provide detailed answers to questions with confidence scores."""
    question: str = dspy.InputField(desc="User's question")
    context: str = dspy.InputField(desc="Additional context", default="")
    answer: str = dspy.OutputField(desc="Detailed answer")
    confidence: float = dspy.OutputField(desc="Confidence from 0-1")

detailed_qa = dspy.Predict(DetailedQA)
```

## 🐛 Troubleshooting

### Common Issues

#### API Key Issues
```python
# Check if API key is loaded
import os
print("OpenAI API Key:", os.getenv('OPENAI_API_KEY', 'Not found'))

# Set API key in code (not recommended for production)
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

#### Import Errors
```bash
# Reinstall with dependencies
pip uninstall dspy-ai
pip install dspy-ai[all]

# Or install specific dependencies
pip install openai transformers sentence-transformers
```

#### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3
```

#### Module Not Found Errors
```python
# Check DSPy installation
import dspy
print(f"DSPy version: {dspy.__version__}")
print(f"DSPy location: {dspy.__file__}")
```

### Getting Help

1. **Documentation**: [Official DSPy Docs](https://dspy.ai/learn/)
2. **GitHub Issues**: [DSPy Repository](https://github.com/stanfordnlp/dspy)
3. **Community**: [Discord Server](https://discord.gg/dspy)
4. **This Guide**: [Troubleshooting Section](13_troubleshooting.md)

## ✅ Installation Checklist

- [ ] Python 3.9+ installed
- [ ] DSPy installed via pip
- [ ] API keys configured (or local model running)
- [ ] Hello World example runs successfully
- [ ] Test script passes
- [ ] Repository examples work

Congratulations! You now have DSPy running. Ready to build your first pipeline?

---

**Next:** [Composing Pipelines: RAG & Summarization →](04_pipelines.md)