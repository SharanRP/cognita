from typing import List, Dict, Any, Optional
from pydantic import Field
from backend.types import ConfiguredBaseModel


class EvaluationDataset(ConfiguredBaseModel):
    """Dataset for RAG evaluation"""

    questions: List[str] = Field(title="List of questions to evaluate")
    ground_truth_answers: Optional[List[str]] = Field(
        None, title="Ground truth answers for the questions"
    )
    contexts: Optional[List[List[str]]] = Field(
        None, title="Expected relevant contexts for each question"
    )


class EvaluationResult(ConfiguredBaseModel):
    """Result of a single evaluation"""

    question: str
    generated_answer: str
    retrieved_contexts: List[str]
    ground_truth_answer: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class EvaluationReport(ConfiguredBaseModel):
    """Complete evaluation report"""

    retriever_name: str
    retriever_config: Dict[str, Any]
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float] = Field(default_factory=dict)
    total_questions: int
    evaluation_time: float


class RetrieverComparison(ConfiguredBaseModel):
    """Comparison between multiple retrievers"""

    collection_name: str
    dataset: EvaluationDataset
    reports: List[EvaluationReport]
    best_retriever: Optional[str] = None
    comparison_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
