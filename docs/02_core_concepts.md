# Core Concepts: Signatures, Modules & Optimizers

DSPy's architecture revolves around three fundamental abstractions that enable **declarative programming** for LLM applications. Understanding these concepts is crucial for building robust, self-improving systems.

## 🎯 Signatures: Declarative Task Specification

A **signature** defines the input/output behavior of a task without specifying how the language model should accomplish it. Think of it as a **type declaration** or **API contract** for your LLM interactions.

### Basic Signature Syntax

```python
import dspy

# String-based signature (simple)
qa_signature = "question -> answer"

# Class-based signature (detailed)
class QuestionAnswering(dspy.Signature):
    """Answer questions accurately based on given context."""
    question: str = dspy.InputField(desc="User's question")
    context: str = dspy.InputField(desc="Relevant background information")
    answer: str = dspy.OutputField(desc="Accurate, concise answer")
```

### Signature Components

#### Input Fields
- Define what data your module expects
- Include optional descriptions for better prompting
- Support type hints for validation

```python
class DocumentAnalysis(dspy.Signature):
    """Analyze documents for key insights."""
    document: str = dspy.InputField(desc="Raw document text")
    focus_area: str = dspy.InputField(desc="Specific area to analyze")
    max_length: int = dspy.InputField(desc="Maximum response length")
```

#### Output Fields
- Specify expected outputs with clear descriptions
- Support multiple outputs for complex tasks
- Enable structured data extraction

```python
class RiskAssessment(dspy.Signature):
    """Assess risk levels in business documents."""
    document: str = dspy.InputField()
    
    # Multiple outputs
    risk_score: int = dspy.OutputField(desc="Risk score from 1-10")
    risk_factors: str = dspy.OutputField(desc="Key risk factors identified")
    mitigation: str = dspy.OutputField(desc="Recommended mitigation strategies")
```

### Advanced Signature Features

#### Constraints and Validation
```python
class ClassificationTask(dspy.Signature):
    """Classify text into predefined categories."""
    text: str = dspy.InputField()
    
    # Constrained output with specific format
    category: str = dspy.OutputField(
        desc="One of: positive, negative, neutral",
        format="single_word"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0 and 1",
        constraints="0 <= confidence <= 1"
    )
```

## 🧩 Modules: Composable Building Blocks

A **module** encapsulates a particular prompting strategy or reasoning pattern. DSPy provides several built-in modules, and you can create custom ones by subclassing `dspy.Module`.

### Built-in Modules

#### 1. Predict (Basic Prompting)
```python
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")
    
    def forward(self, question: str):
        return self.predictor(question=question)
```

#### 2. ChainOfThought (Step-by-Step Reasoning)
```python
class ReasoningQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str):
        # Automatically includes reasoning steps
        return self.cot(question=question)
```

#### 3. ProgramOfThought (Code Generation)
```python
class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pot = dspy.ProgramOfThought("math_problem -> solution")
    
    def forward(self, math_problem: str):
        # Generates and executes code to solve problems
        return self.pot(math_problem=math_problem)
```

#### 4. ReAct (Reasoning + Acting with Tools)
```python
def calculator(expression: str) -> float:
    """Safe calculator function."""
    return eval(expression)

class CalculatorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("question -> answer", tools=[calculator])
    
    def forward(self, question: str):
        # Can call tools during reasoning
        return self.react(question=question)
```

### Custom Modules

Create sophisticated pipelines by composing multiple modules:

```python
class AdvancedRAG(dspy.Module):
    """Multi-step retrieval-augmented generation."""
    
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        
        # Compose multiple modules
        self.query_rewriter = dspy.ChainOfThought("question -> rewritten_query")
        self.retriever = dspy.Retrieve(k=self.k)
        self.reranker = dspy.Predict("query, contexts -> ranked_contexts")
        self.generator = dspy.ChainOfThought("query, context -> answer")
    
    def forward(self, question: str):
        # Multi-step pipeline
        rewritten = self.query_rewriter(question=question)
        contexts = self.retriever(rewritten.rewritten_query)
        ranked = self.reranker(query=question, contexts=contexts)
        answer = self.generator(query=question, context=ranked.ranked_contexts)
        
        return dspy.Prediction(
            answer=answer.answer,
            reasoning=answer.reasoning,
            contexts=contexts,
            rewritten_query=rewritten.rewritten_query
        )
```

### Module Composition Patterns

#### Sequential Composition
```python
class Pipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.Predict("intermediate -> output")
    
    def forward(self, input_data):
        intermediate = self.step1(input=input_data)
        output = self.step2(intermediate=intermediate.intermediate)
        return output
```

#### Parallel Composition
```python
class ParallelProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer1 = dspy.Predict("text -> sentiment")
        self.analyzer2 = dspy.Predict("text -> topics")
        self.combiner = dspy.ChainOfThought("sentiment, topics -> summary")
    
    def forward(self, text):
        # Parallel processing
        sentiment = self.analyzer1(text=text)
        topics = self.analyzer2(text=text)
        
        # Combine results
        summary = self.combiner(
            sentiment=sentiment.sentiment,
            topics=topics.topics
        )
        return summary
```

## 🚀 Optimizers: Self-Improving Pipelines

**Optimizers** (also called teleprompters) automatically improve your modules by generating better prompts and few-shot examples. They use example data and metrics to iteratively enhance performance.

### BootstrapFewShot Optimizer

The most commonly used optimizer that creates effective few-shot examples:

```python
from dspy.teleprompt import BootstrapFewShot
from dspy import metrics, Example

# Prepare training data
train_examples = [
    Example(
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("question"),
    Example(
        question="Who wrote Romeo and Juliet?",
        answer="William Shakespeare"
    ).with_inputs("question"),
    # ... more examples
]

# Create and configure optimizer
teleprompter = BootstrapFewShot(
    metric=metrics.answer_exact_match,
    max_bootstrapped_demos=5,  # Number of examples to generate
    max_labeled_demos=3        # Number of labeled examples to use
)

# Original module
qa_module = SimpleQA()

# Optimized version
optimized_qa = teleprompter.compile(qa_module, trainset=train_examples)
```

### COPRO (Coordinate Ascent Prompt Optimization)
```python
from dspy.teleprompt import COPRO

# Advanced prompt optimization
copro = COPRO(
    metric=metrics.answer_exact_match,
    breadth=10,  # Number of prompt candidates to try
    depth=3      # Optimization iterations
)

optimized_module = copro.compile(module, trainset=examples)
```

### BetterTogether (Multi-Module Optimization)
```python
from dspy.teleprompt import BetterTogether

# Jointly optimize multiple modules
pipeline = ComplexPipeline()  # Module with multiple sub-modules
optimizer = BetterTogether(metric=custom_metric)
optimized_pipeline = optimizer.compile(pipeline, trainset=examples)
```

### Custom Metrics

Create domain-specific metrics for optimization:

```python
def semantic_similarity_metric(prediction, ground_truth):
    """Custom metric using sentence embeddings."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    pred_embedding = model.encode([prediction.answer])
    true_embedding = model.encode([ground_truth.answer])
    
    # Cosine similarity
    similarity = np.dot(pred_embedding[0], true_embedding[0]) / (
        np.linalg.norm(pred_embedding[0]) * np.linalg.norm(true_embedding[0])
    )
    
    return max(0, similarity)  # Ensure non-negative score

# Use custom metric in optimization
optimizer = BootstrapFewShot(metric=semantic_similarity_metric)
```

## 🔄 Putting It All Together

Here's how signatures, modules, and optimizers work together:

```python
import dspy
from dspy import Example, Evaluate, metrics
from dspy.teleprompt import BootstrapFewShot

# 1. Define signature (what we want to accomplish)
class SentimentAnalysis(dspy.Signature):
    """Analyze sentiment in customer reviews."""
    review: str = dspy.InputField(desc="Customer review text")
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")

# 2. Create module (how to accomplish it)
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SentimentAnalysis)
    
    def forward(self, review: str):
        return self.analyze(review=review)

# 3. Prepare data for optimization
train_data = [
    Example(review="This product is amazing!", sentiment="positive", confidence=0.9),
    Example(review="Terrible quality, waste of money", sentiment="negative", confidence=0.95),
    # ... more examples
]

# 4. Optimize the module
analyzer = SentimentAnalyzer()
optimizer = BootstrapFewShot(metric=metrics.answer_exact_match)
optimized_analyzer = optimizer.compile(analyzer, trainset=train_data)

# 5. Evaluate performance
evaluator = Evaluate(testset=test_data, metric=metrics.answer_exact_match)
baseline_score = evaluator(analyzer)
optimized_score = evaluator(optimized_analyzer)

print(f"Baseline: {baseline_score:.2%}")
print(f"Optimized: {optimized_score:.2%}")
```

## 🎯 Key Takeaways

1. **Signatures** provide type safety and clear contracts
2. **Modules** encapsulate reusable prompting strategies
3. **Optimizers** automatically improve performance using data
4. **Composition** enables building complex pipelines from simple parts
5. **Evaluation** ensures systematic performance measurement

Understanding these three concepts is essential for building production-ready DSPy applications. Next, we'll walk through installation and create your first DSPy program.

---

**Next:** [Installation & Hello World →](03_installation.md)