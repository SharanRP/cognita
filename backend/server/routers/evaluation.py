from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.logger import logger
from backend.modules.evaluation.evaluator import RAGEvaluator
from backend.modules.evaluation.types import EvaluationDataset
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_controllers.example.controller import BasicRAGQueryController
from backend.types import ModelConfig

router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])


class CompareRetrieversRequest(BaseModel):
    """Request model for comparing retrievers"""

    collection_name: str
    questions: List[str]
    retriever_configs: List[Dict[str, Any]]
    llm_config: Dict[str, Any]  # Renamed from model_config
    ground_truth_answers: Optional[List[str]] = None
    prompt_template: str = (
        "Answer the question based on the context: {context}\n\nQuestion: {question}"
    )
    evaluation_model_name: Optional[str] = None


class EvaluateRetrieverRequest(BaseModel):
    """Request model for evaluating single retriever"""

    collection_name: str
    retriever_name: str
    retriever_config: Dict[str, Any]
    questions: List[str]
    llm_config: Dict[str, Any]  # Renamed from model_config
    ground_truth_answers: Optional[List[str]] = None
    prompt_template: str = (
        "Answer the question based on the context: {context}\n\nQuestion: {question}"
    )
    evaluation_model_name: Optional[str] = None


@router.post("/compare-retrievers")
async def compare_retrievers(request: CompareRetrieversRequest):
    """
    Compare multiple retriever configurations on a dataset

    Args:
        collection_name: Name of the collection to evaluate on
        questions: List of questions to evaluate
        retriever_configs: List of retriever configurations to compare
        model_config: LLM model configuration for generating answers
        ground_truth_answers: Optional ground truth answers for comparison
        prompt_template: Template for generating answers
        evaluation_model_name: Optional model name for evaluation metrics
    """
    try:
        logger.info(
            f"Starting retriever comparison for collection: {request.collection_name}"
        )

        # Create dataset
        dataset = EvaluationDataset(
            questions=request.questions,
            ground_truth_answers=request.ground_truth_answers,
        )

        # Create model config
        model_cfg = ModelConfig(**request.llm_config)

        # Get evaluation LLM if specified
        evaluation_llm = None
        if request.evaluation_model_name:
            try:
                evaluation_llm = model_gateway.get_llm_from_model_config(
                    ModelConfig(name=request.evaluation_model_name)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load evaluation model {request.evaluation_model_name}: {e}"
                )

        # Create evaluator and query controller
        evaluator = RAGEvaluator()
        query_controller = BasicRAGQueryController()

        # Run comparison
        comparison = await evaluator.compare_retrievers(
            collection_name=request.collection_name,
            retriever_configs=request.retriever_configs,
            dataset=dataset,
            model_config=model_cfg,
            prompt_template=request.prompt_template,
            query_controller=query_controller,
            evaluation_llm=evaluation_llm,
        )

        return JSONResponse(content=comparison.model_dump(), status_code=200)

    except Exception as e:
        logger.error(f"Error in retriever comparison: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to compare retrievers: {str(e)}"
        )


@router.post("/evaluate-retriever")
async def evaluate_single_retriever(request: EvaluateRetrieverRequest):
    """
    Evaluate a single retriever configuration

    Args:
        collection_name: Name of the collection to evaluate on
        retriever_name: Name of the retriever to evaluate
        retriever_config: Configuration for the retriever
        questions: List of questions to evaluate
        model_config: LLM model configuration for generating answers
        ground_truth_answers: Optional ground truth answers for comparison
        prompt_template: Template for generating answers
        evaluation_model_name: Optional model name for evaluation metrics
    """
    try:
        logger.info(f"Starting evaluation for retriever: {request.retriever_name}")

        # Create dataset
        dataset = EvaluationDataset(
            questions=request.questions,
            ground_truth_answers=request.ground_truth_answers,
        )

        # Create model config
        model_cfg = ModelConfig(**request.llm_config)

        # Get evaluation LLM if specified
        evaluation_llm = None
        if request.evaluation_model_name:
            try:
                evaluation_llm = model_gateway.get_llm_from_model_config(
                    ModelConfig(name=request.evaluation_model_name)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load evaluation model {request.evaluation_model_name}: {e}"
                )

        # Create evaluator and query controller
        evaluator = RAGEvaluator()
        query_controller = BasicRAGQueryController()

        # Run evaluation
        report = await evaluator.evaluate_retriever(
            collection_name=request.collection_name,
            retriever_name=request.retriever_name,
            retriever_config=request.retriever_config,
            dataset=dataset,
            model_config=model_cfg,
            prompt_template=request.prompt_template,
            query_controller=query_controller,
            evaluation_llm=evaluation_llm,
        )

        return JSONResponse(content=report.model_dump(), status_code=200)

    except Exception as e:
        logger.error(f"Error in retriever evaluation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to evaluate retriever: {str(e)}"
        )


@router.get("/metrics")
async def get_available_metrics():
    """Get list of available evaluation metrics"""
    metrics = {
        "context_relevance": "Measures how relevant retrieved contexts are to the question",
        "answer_relevance": "Measures how relevant the generated answer is to the question",
        "faithfulness": "Measures how faithful the answer is to the provided contexts",
        "ground_truth_similarity": "Measures similarity between generated answer and ground truth",
    }

    return JSONResponse(content={"metrics": metrics})


@router.get("/retrievers")
async def get_available_retrievers():
    """Get list of available retriever types"""
    retrievers = {
        "vectorstore": "Basic vector similarity search",
        "multi-query": "Multi-query retrieval with query expansion",
        "contextual-compression": "Retrieval with context compression/reranking",
        "contextual-compression-multi-query": "Combined multi-query and compression",
    }

    return JSONResponse(content={"retrievers": retrievers})
