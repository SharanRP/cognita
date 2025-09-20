import React, { useState } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    Typography,
    TextField,
    Grid,
    Chip,
    CircularProgress,
    Alert,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper
} from '@mui/material';
import { faChevronDown } from '@fortawesome/free-solid-svg-icons';
import IconProvider from '@/components/assets/IconProvider';
import { evaluateRetrievers } from '../api/evaluation';

interface EvaluationResult {
    retriever_name: string;
    aggregate_metrics: Record<string, number>;
    total_questions: number;
    evaluation_time: number;
}

interface ComparisonResult {
    best_retriever: string;
    reports: EvaluationResult[];
}

const EvaluationScreen: React.FC = () => {
    const [collectionName, setCollectionName] = useState('');
    const [questions, setQuestions] = useState('');
    const [groundTruthAnswers, setGroundTruthAnswers] = useState('');
    const [modelName, setModelName] = useState('local-ollama/qwen2:1.5b');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<ComparisonResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const defaultRetrieverConfigs = [
        {
            name: 'vectorstore',
            config: {
                search_type: 'similarity',
                search_kwargs: { k: 4 }
            }
        },
        {
            name: 'multi-query',
            config: {
                search_type: 'similarity',
                search_kwargs: { k: 4 },
                retriever_llm_configuration: {
                    name: 'local-ollama/qwen2:1.5b'
                }
            }
        },
        {
            name: 'contextual-compression',
            config: {
                search_type: 'similarity',
                search_kwargs: { k: 8 },
                compressor_model_name: 'local-infinity/mixedbread-ai/mxbai-rerank-xsmall-v1',
                top_k: 4
            }
        }
    ];

    const handleEvaluate = async () => {
        if (!collectionName.trim() || !questions.trim()) {
            setError('Please provide collection name and questions');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const questionList = questions.split('\n').filter(q => q.trim());
            const groundTruthList = groundTruthAnswers
                ? groundTruthAnswers.split('\n').filter(a => a.trim())
                : undefined;

            const response = await evaluateRetrievers({
                collection_name: collectionName,
                questions: questionList,
                retriever_configs: defaultRetrieverConfigs,
                llm_config: { name: modelName },
                ground_truth_answers: groundTruthList
            });

            setResults(response);
        } catch (err: any) {
            setError(err.message || 'Failed to run evaluation');
        } finally {
            setLoading(false);
        }
    };

    const formatMetricName = (metricName: string) => {
        return metricName
            .replace('_avg', '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    };

    const getMetricColor = (score: number) => {
        if (score >= 0.8) return 'success';
        if (score >= 0.6) return 'warning';
        return 'error';
    };

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
                RAG Evaluation
            </Typography>

            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Compare different retriever configurations to find the best performing setup for your collection.
            </Typography>

            <Grid container spacing={3}>
                {/* Configuration Panel */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Evaluation Configuration
                            </Typography>

                            <TextField
                                fullWidth
                                label="Collection Name"
                                value={collectionName}
                                onChange={(e) => setCollectionName(e.target.value)}
                                margin="normal"
                                required
                            />

                            <TextField
                                fullWidth
                                label="Model Name"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                margin="normal"
                                helperText="Model to use for generating answers"
                            />

                            <TextField
                                fullWidth
                                multiline
                                rows={6}
                                label="Questions (one per line)"
                                value={questions}
                                onChange={(e) => setQuestions(e.target.value)}
                                margin="normal"
                                required
                                placeholder="What is machine learning?&#10;How does AI work?&#10;..."
                            />

                            <TextField
                                fullWidth
                                multiline
                                rows={4}
                                label="Ground Truth Answers (optional, one per line)"
                                value={groundTruthAnswers}
                                onChange={(e) => setGroundTruthAnswers(e.target.value)}
                                margin="normal"
                                placeholder="Machine learning is...&#10;AI works by...&#10;..."
                            />

                            <Box sx={{ mt: 2 }}>
                                <Typography variant="subtitle2" gutterBottom>
                                    Retrievers to Compare:
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                    {defaultRetrieverConfigs.map((config) => (
                                        <Chip key={config.name} label={config.name} variant="outlined" />
                                    ))}
                                </Box>
                            </Box>

                            <Button
                                fullWidth
                                variant="contained"
                                onClick={handleEvaluate}
                                disabled={loading}
                                sx={{ mt: 3 }}
                            >
                                {loading ? <CircularProgress size={24} /> : 'Start Evaluation'}
                            </Button>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Results Panel */}
                <Grid item xs={12} md={6}>
                    {error && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {error}
                        </Alert>
                    )}

                    {results && (
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Evaluation Results
                                </Typography>

                                <Alert severity="success" sx={{ mb: 2 }}>
                                    Best Retriever: <strong>{results.best_retriever}</strong>
                                </Alert>

                                <TableContainer component={Paper} variant="outlined">
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Retriever</TableCell>
                                                <TableCell align="right">Questions</TableCell>
                                                <TableCell align="right">Time (s)</TableCell>
                                                <TableCell align="right">Avg Score</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {results.reports.map((report) => {
                                                const avgMetrics = Object.entries(report.aggregate_metrics)
                                                    .filter(([key]) => key.endsWith('_avg'))
                                                    .map(([, value]) => value);
                                                const overallScore = avgMetrics.length > 0
                                                    ? avgMetrics.reduce((a, b) => a + b, 0) / avgMetrics.length
                                                    : 0;

                                                return (
                                                    <TableRow key={report.retriever_name}>
                                                        <TableCell>{report.retriever_name}</TableCell>
                                                        <TableCell align="right">{report.total_questions}</TableCell>
                                                        <TableCell align="right">{report.evaluation_time.toFixed(1)}</TableCell>
                                                        <TableCell align="right">
                                                            <Chip
                                                                label={overallScore.toFixed(3)}
                                                                color={getMetricColor(overallScore)}
                                                                size="small"
                                                            />
                                                        </TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </TableContainer>

                                {/* Detailed Metrics */}
                                <Box sx={{ mt: 2 }}>
                                    {results.reports.map((report) => (
                                        <Accordion key={report.retriever_name}>
                                            <AccordionSummary expandIcon={<IconProvider icon={faChevronDown} />}>
                                                <Typography variant="subtitle1">
                                                    {report.retriever_name} - Detailed Metrics
                                                </Typography>
                                            </AccordionSummary>
                                            <AccordionDetails>
                                                <Grid container spacing={2}>
                                                    {Object.entries(report.aggregate_metrics)
                                                        .filter(([key]) => key.endsWith('_avg'))
                                                        .map(([key, value]) => (
                                                            <Grid item xs={6} key={key}>
                                                                <Box sx={{ textAlign: 'center' }}>
                                                                    <Typography variant="body2" color="text.secondary">
                                                                        {formatMetricName(key)}
                                                                    </Typography>
                                                                    <Chip
                                                                        label={value.toFixed(3)}
                                                                        color={getMetricColor(value)}
                                                                        variant="outlined"
                                                                    />
                                                                </Box>
                                                            </Grid>
                                                        ))}
                                                </Grid>
                                            </AccordionDetails>
                                        </Accordion>
                                    ))}
                                </Box>
                            </CardContent>
                        </Card>
                    )}
                </Grid>
            </Grid>
        </Box>
    );
};

export default EvaluationScreen;