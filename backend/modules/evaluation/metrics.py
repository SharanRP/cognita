import re
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel


class BaseMetric(ABC):
    """Base class for evaluation metrics"""

    @abstractmethod
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> float:
        """Calculate the metric score"""
        pass


class ContextRelevanceMetric(BaseMetric):
    """Measures how relevant the retrieved contexts are to the question"""

    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> float:
        if not contexts or not llm:
            return 0.0

        relevant_count = 0
        for context in contexts:
            prompt = f"""
            Question: {question}
            Context: {context}
            
            Is this context relevant to answering the question? Answer only 'Yes' or 'No'.
            """

            try:
                response = llm.invoke(prompt)
                if hasattr(response, "content"):
                    response_text = response.content.strip().lower()
                else:
                    response_text = str(response).strip().lower()

                if "yes" in response_text:
                    relevant_count += 1
            except Exception:
                # If LLM call fails, assume context is relevant
                relevant_count += 1

        return relevant_count / len(contexts) if contexts else 0.0


class AnswerRelevanceMetric(BaseMetric):
    """Measures how relevant the generated answer is to the question"""

    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> float:
        if not answer or not llm:
            return 0.0

        prompt = f"""
        Question: {question}
        Answer: {answer}
        
        Rate how relevant this answer is to the question on a scale of 0-10, where:
        - 0 = Completely irrelevant
        - 5 = Somewhat relevant
        - 10 = Perfectly relevant
        
        Respond with only a number between 0 and 10.
        """

        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()

            # Extract number from response
            numbers = re.findall(r"\d+", response_text)
            if numbers:
                score = min(10, max(0, int(numbers[0])))
                return score / 10.0
        except Exception:
            pass

        return 0.5  # Default neutral score


class FaithfulnessMetric(BaseMetric):
    """Measures how faithful the answer is to the provided contexts"""

    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> float:
        if not answer or not contexts or not llm:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""
        Context: {context_text}
        Answer: {answer}
        
        Is the answer faithful to the provided context? Does it contain information not supported by the context?
        Rate faithfulness on a scale of 0-10, where:
        - 0 = Answer contradicts or adds unsupported information
        - 5 = Answer is partially supported by context
        - 10 = Answer is fully supported by context
        
        Respond with only a number between 0 and 10.
        """

        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()

            # Extract number from response
            numbers = re.findall(r"\d+", response_text)
            if numbers:
                score = min(10, max(0, int(numbers[0])))
                return score / 10.0
        except Exception:
            pass

        return 0.5  # Default neutral score


class GroundTruthMetric(BaseMetric):
    """Measures similarity between generated answer and ground truth"""

    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> float:
        if not answer or not ground_truth or not llm:
            return 0.0

        prompt = f"""
        Ground Truth Answer: {ground_truth}
        Generated Answer: {answer}
        
        Rate how similar these answers are on a scale of 0-10, where:
        - 0 = Completely different
        - 5 = Somewhat similar
        - 10 = Essentially the same
        
        Respond with only a number between 0 and 10.
        """

        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()

            # Extract number from response
            numbers = re.findall(r"\d+", response_text)
            if numbers:
                score = min(10, max(0, int(numbers[0])))
                return score / 10.0
        except Exception:
            pass

        return 0.0  # Default to no similarity if can't evaluate
