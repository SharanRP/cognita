import { instance as apiClient } from './utils';

export interface EvaluationRequest {
  collection_name: string;
  questions: string[];
  retriever_configs: Array<{
    name: string;
    config: Record<string, any>;
  }>;
  llm_config: {  // Changed from model_config
    name: string;
  };
  ground_truth_answers?: string[];
  prompt_template?: string;
  evaluation_model_name?: string;
}

export interface EvaluationResult {
  retriever_name: string;
  aggregate_metrics: Record<string, number>;
  total_questions: number;
  evaluation_time: number;
  results: Array<{
    question: string;
    generated_answer: string;
    retrieved_contexts: string[];
    ground_truth_answer?: string;
    metrics: Record<string, number>;
  }>;
}

export interface ComparisonResult {
  collection_name: string;
  best_retriever: string;
  reports: EvaluationResult[];
  comparison_metrics: Record<string, Record<string, number>>;
}

export const evaluateRetrievers = async (
  request: EvaluationRequest
): Promise<ComparisonResult> => {
  const response = await apiClient.post('/v1/evaluation/compare-retrievers', request);
  return response.data;
};

export const evaluateSingleRetriever = async (
  collectionName: string,
  retrieverName: string,
  retrieverConfig: Record<string, any>,
  questions: string[],
  llmConfig: { name: string },
  groundTruthAnswers?: string[],
  promptTemplate?: string,
  evaluationModelName?: string
): Promise<EvaluationResult> => {
  const response = await apiClient.post('/v1/evaluation/evaluate-retriever', {
    collection_name: collectionName,
    retriever_name: retrieverName,
    retriever_config: retrieverConfig,
    questions,
    llm_config: llmConfig,
    ground_truth_answers: groundTruthAnswers,
    prompt_template: promptTemplate,
    evaluation_model_name: evaluationModelName,
  });
  return response.data;
};

export const getAvailableMetrics = async (): Promise<Record<string, string>> => {
  const response = await apiClient.get('/v1/evaluation/metrics');
  return response.data.metrics;
};

export const getAvailableRetrievers = async (): Promise<Record<string, string>> => {
  const response = await apiClient.get('/v1/evaluation/retrievers');
  return response.data.retrievers;
};