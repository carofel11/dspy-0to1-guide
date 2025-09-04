#!/usr/bin/env python3
"""
Test Suite for Basic DSPy Examples

Tests the core functionality of all basic examples.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def configure_dspy():
    """Configure DSPy for all tests."""
    try:
        if os.getenv('OPENAI_API_KEY'):
            dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
        else:
            dspy.configure(lm=dspy.LM('ollama_chat/llama3'))
        yield
    except Exception as e:
        pytest.skip(f"Could not configure DSPy: {e}")

class TestBasicExamples:
    """Test basic DSPy functionality."""
    
    def test_hello_world_qa(self):
        """Test basic question answering."""
        import dspy
        
        # Simple direct test since hello_world.py doesn't export classes
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is the capital of France?")
        
        assert hasattr(result, 'answer')
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
    
    def test_math_qa_basic(self):
        """Test mathematical reasoning."""
        from examples.basic.math_qa import MathQA
        
        math_qa = MathQA()
        result = math_qa("What is 2 + 2?")
        
        assert hasattr(result, 'solution')
        # Should contain numeric answer
        assert any(char.isdigit() for char in str(result.solution))
    
    def test_summarizer_basic(self):
        """Test document summarization."""
        from examples.basic.summarizer import DocumentSummarizer
        
        summarizer = DocumentSummarizer()
        document = "This is a test document with multiple sentences. It contains information about testing. The document is used to verify summarization functionality."
        
        result = summarizer(document, max_length=50)
        
        assert hasattr(result, 'summary')
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert len(result.summary.split()) < len(document.split())

class TestPersonaExamples:
    """Test persona-driven examples."""
    
    @pytest.mark.slow
    def test_support_sam_ticket_processing(self):
        """Test Support-Sam ticket processing."""
        from examples.personas.support_sam import SupportSam
        
        sam = SupportSam()
        ticket = "I can't log into my account. Getting invalid credentials error."
        
        result = sam.process_ticket(ticket, "TEST001")
        
        assert isinstance(result, dict)
        assert 'ticket_id' in result
        assert 'classification' in result
        assert 'response' in result
        assert result['classification']['category'] in ['technical', 'billing', 'account', 'product', 'general']
        assert result['classification']['urgency'] in ['low', 'medium', 'high', 'critical']

class TestAdvancedExamples:
    """Test advanced DSPy features."""
    
    def test_pydantic_validation_structure(self):
        """Test Pydantic validation creates proper structures."""
        from examples.advanced.pydantic_validation import StructuredContractAnalyzer, ContractAnalysisResult
        
        analyzer = StructuredContractAnalyzer()
        contract = "This is a service agreement between Company A and Company B for consulting services."
        
        result = analyzer(contract)
        
        assert isinstance(result, ContractAnalysisResult)
        assert hasattr(result, 'contract_type')
        assert hasattr(result, 'parties')
        assert hasattr(result, 'risk_level')
        assert result.compliance_score >= 0
        assert result.compliance_score <= 100

class TestDatasets:
    """Test dataset functionality."""
    
    def test_qa_dataset_format(self):
        """Test QA dataset has proper format."""
        import json
        
        qa_path = project_root / "datasets/sample_data/qa.json"
        if qa_path.exists():
            with open(qa_path) as f:
                data = json.load(f)
            
            assert 'data' in data
            assert len(data['data']) > 0
            
            example = data['data'][0]
            assert 'question' in example
            assert 'answer' in example
            assert isinstance(example['question'], str)
            assert isinstance(example['answer'], str)
    
    def test_dspy_examples_format(self):
        """Test DSPy examples format."""
        import json
        
        dspy_qa_path = project_root / "datasets/sample_data/dspy_qa_examples.json"
        if dspy_qa_path.exists():
            with open(dspy_qa_path) as f:
                data = json.load(f)
            
            assert 'data' in data
            assert len(data['data']) > 0
            
            example = data['data'][0]
            assert 'question' in example
            assert 'answer' in example

class TestInfrastructure:
    """Test infrastructure components."""
    
    def test_prometheus_metrics_collector(self):
        """Test Prometheus metrics collector."""
        from examples.infrastructure.prometheus_metrics import DSPyMetricsCollector
        
        collector = DSPyMetricsCollector()
        
        # Test metric recording
        collector.record_llm_request(
            model="test-model",
            module_type="qa",
            duration=0.5,
            status="success",
            input_tokens=10,
            output_tokens=5
        )
        
        # Check that metrics were recorded (basic test)
        assert collector.llm_requests._value._value > 0
    
    def test_instrumented_module_base(self):
        """Test instrumented module base class."""
        from examples.infrastructure.prometheus_metrics import DSPyMetricsCollector, InstrumentedModule
        
        collector = DSPyMetricsCollector()
        
        class TestModule(InstrumentedModule):
            def _forward_impl(self, test_input):
                return {"result": f"processed: {test_input}"}
        
        module = TestModule(collector, "test")
        result = module("test input")
        
        assert result is not None
        assert isinstance(result, dict)

@pytest.mark.slow
class TestIntegration:
    """Integration tests that may take longer."""
    
    def test_end_to_end_qa_pipeline(self):
        """Test complete QA pipeline."""
        # Simple QA test
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is 1 + 1?")
        
        assert hasattr(result, 'answer')
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
    
    def test_chain_of_thought_reasoning(self):
        """Test chain of thought reasoning."""
        cot = dspy.ChainOfThought("math_problem -> solution")
        result = cot(math_problem="If I have 5 apples and give away 2, how many do I have left?")
        
        assert hasattr(result, 'solution')
        assert isinstance(result.solution, str)
        # Should mention 3 as the answer
        assert "3" in result.solution

class TestConfiguration:
    """Test configuration and setup."""
    
    def test_dspy_configuration(self):
        """Test DSPy is properly configured."""
        assert dspy.settings.lm is not None
    
    def test_environment_variables(self):
        """Test environment setup."""
        # Test that .env file exists and is readable
        env_path = project_root / ".env"
        assert env_path.exists()
        
        # Test basic environment structure
        with open(env_path) as f:
            content = f.read()
            assert "OLLAMA_HOST" in content
            assert "OLLAMA_MODEL" in content