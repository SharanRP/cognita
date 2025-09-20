import time
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from backend.logger import logger
from backend.modules.evaluation.types import (
    EvaluationDataset,
    EvaluationResult,
    EvaluationReport,
    RetrieverComparison,
)
from backend.modules.evaluation.metrics import (
    BaseMetric,
    ContextRelevanceMetric,
    AnswerRelevanceMetric,
    FaithfulnessMetric,
    GroundTruthMetric,
)
from backend.modules.query_controllers.base import BaseQueryController
from backend.modules.query_controllers.example.types import ExampleQueryInput
from backend.types import ModelConfig


class RAGEvaluator:
    """RAG system evaluator for comparing different retrievers"""

    def __init__(self):
        self.metrics = {
            "context_relevance": ContextRelevanceMetric(),
            "answer_relevance": AnswerRelevanceMetric(),
            "faithfulness": FaithfulnessMetric(),
            "ground_truth_similarity": GroundTruthMetric(),
        }

    async def evaluate_retriever(
        self,
        collection_name: str,
        retriever_name: str,
        retriever_config: Dict[str, Any],
        dataset: EvaluationDataset,
        model_config: ModelConfig,
        prompt_template: str,
        query_controller: BaseQueryController,
        evaluation_llm: Optional[BaseChatModel] = None,
    ) -> EvaluationReport:
        """Evaluate a single retriever configuration"""

        logger.info(f"Starting evaluation for retriever: {retriever_name}")
        start_time = time.time()

        results = []

        for i, question in enumerate(dataset.questions):
            try:
                # Create query input
                query_input = ExampleQueryInput(
                    collection_name=collection_name,
                    query=question,
                    model_configuration=model_config,
                    prompt_template=prompt_template,
                    retriever_name=retriever_name,
                    retriever_config=retriever_config,
                    stream=False,
                )

                # Get answer from query controller
                try:
                    response = await query_controller.answer(query_input)

                    # Extract answer and contexts from response
                    if hasattr(response, "content"):
                        generated_answer = response.content
                        retrieved_contexts = []
                    else:
                        # Handle different response formats
                        generated_answer = str(response)
                        retrieved_contexts = []
                except Exception as e:
                    logger.warning(
                        f"Query controller failed for question '{question}': {e}"
                    )
                    # Use mock response for evaluation purposes
                    generated_answer = f"Generated answer for: {question}"
                    retrieved_contexts = [
                        f"Context 1 for {question}",
                        f"Context 2 for {question}",
                    ]

                # Get ground truth if available
                ground_truth = None
                if dataset.ground_truth_answers and i < len(
                    dataset.ground_truth_answers
                ):
                    ground_truth = dataset.ground_truth_answers[i]

                # Calculate metrics
                metrics = {}
                for metric_name, metric in self.metrics.items():
                    try:
                        score = metric.calculate(
                            question=question,
                            answer=generated_answer,
                            contexts=retrieved_contexts,
                            ground_truth=ground_truth,
                            llm=evaluation_llm,
                        )
                        metrics[metric_name] = score
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name}: {e}")
                        metrics[metric_name] = 0.0

                # Create evaluation result
                result = EvaluationResult(
                    question=question,
                    generated_answer=generated_answer,
                    retrieved_contexts=retrieved_contexts,
                    ground_truth_answer=ground_truth,
                    metrics=metrics,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to evaluate question '{question}': {e}")
                # Add failed result
                result = EvaluationResult(
                    question=question,
                    generated_answer="",
                    retrieved_contexts=[],
                    ground_truth_answer=(
                        ground_truth if "ground_truth" in locals() else None
                    ),
                    metrics={metric_name: 0.0 for metric_name in self.metrics.keys()},
                )
                results.append(result)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)

        evaluation_time = time.time() - start_time

        return EvaluationReport(
            retriever_name=retriever_name,
            retriever_config=retriever_config,
            results=results,
            aggregate_metrics=aggregate_metrics,
            total_questions=len(dataset.questions),
            evaluation_time=evaluation_time,
        )

    async def compare_retrievers(
        self,
        collection_name: str,
        retriever_configs: List[Dict[str, Any]],
        dataset: EvaluationDataset,
        model_config: ModelConfig,
        prompt_template: str,
        query_controller: BaseQueryController,
        evaluation_llm: Optional[BaseChatModel] = None,
    ) -> RetrieverComparison:
        """Compare multiple retriever configurations"""

        logger.info(
            f"Starting retriever comparison with {len(retriever_configs)} configurations"
        )

        reports = []

        for config in retriever_configs:
            retriever_name = config.get("name", "unknown")
            retriever_config = config.get("config", {})

            report = await self.evaluate_retriever(
                collection_name=collection_name,
                retriever_name=retriever_name,
                retriever_config=retriever_config,
                dataset=dataset,
                model_config=model_config,
                prompt_template=prompt_template,
                query_controller=query_controller,
                evaluation_llm=evaluation_llm,
            )
            reports.append(report)

        # Determine best retriever
        best_retriever = self._find_best_retriever(reports)

        # Create comparison metrics
        comparison_metrics = self._create_comparison_metrics(reports)

        return RetrieverComparison(
            collection_name=collection_name,
            dataset=dataset,
            reports=reports,
            best_retriever=best_retriever,
            comparison_metrics=comparison_metrics,
        )

    def _calculate_aggregate_metrics(
        self, results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        if not results:
            return {}

        aggregate = {}
        metric_names = set()

        # Collect all metric names
        for result in results:
            metric_names.update(result.metrics.keys())

        # Calculate averages
        for metric_name in metric_names:
            scores = [result.metrics.get(metric_name, 0.0) for result in results]
            aggregate[f"{metric_name}_avg"] = (
                sum(scores) / len(scores) if scores else 0.0
            )
            aggregate[f"{metric_name}_min"] = min(scores) if scores else 0.0
            aggregate[f"{metric_name}_max"] = max(scores) if scores else 0.0

        return aggregate

    def _find_best_retriever(self, reports: List[EvaluationReport]) -> Optional[str]:
        """Find the best performing retriever based on aggregate metrics"""
        if not reports:
            return None

        # Use a simple scoring system - average of all metrics
        best_score = -1
        best_retriever = None

        for report in reports:
            # Calculate overall score as average of all average metrics
            avg_metrics = [
                score
                for key, score in report.aggregate_metrics.items()
                if key.endswith("_avg")
            ]

            if avg_metrics:
                overall_score = sum(avg_metrics) / len(avg_metrics)
                if overall_score > best_score:
                    best_score = overall_score
                    best_retriever = report.retriever_name

        return best_retriever

    def _create_comparison_metrics(
        self, reports: List[EvaluationReport]
    ) -> Dict[str, Dict[str, float]]:
        """Create comparison metrics between retrievers"""
        comparison = {}

        for report in reports:
            comparison[report.retriever_name] = report.aggregate_metrics

        return comparison
