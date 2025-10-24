'use client';

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { ImputationDataPoint } from '@/types/imputation';
import { getMethodColor } from '@/utils/colors';

interface BenchmarkLossChartsProps {
  data: ImputationDataPoint[];
}

export default function BenchmarkLossCharts({ data }: BenchmarkLossChartsProps) {
  // Filter for benchmark_loss data
  const benchmarkData = useMemo(() => {
    return data.filter(d => d.type === 'benchmark_loss');
  }, [data]);

  // State for selected method in train/test comparison
  const [selectedMethod, setSelectedMethod] = useState<string>('');

  // Check if we have benchmark loss data
  const hasBenchmarkData = benchmarkData.length > 0;

  // Separate quantile loss and log loss data
  const { quantileLossData, logLossData, methods } = useMemo(() => {
    const quantile = benchmarkData.filter(
      d => d.metric_name === 'quantile_loss' &&
           d.split === 'test' &&
           typeof d.quantile === 'number' &&
           d.quantile >= 0 &&
           d.quantile <= 1
    );

    const logLoss = benchmarkData.filter(
      d => d.metric_name === 'log_loss' &&
           d.split === 'test' &&
           d.metric_value !== null
    );

    // Get unique methods
    const uniqueMethods = Array.from(new Set(benchmarkData.map(d => d.method)));

    return {
      quantileLossData: quantile,
      logLossData: logLoss,
      methods: uniqueMethods,
    };
  }, [benchmarkData]);

  // Transform quantile loss data for grouped bar chart
  const quantileChartData = useMemo(() => {
    if (quantileLossData.length === 0) return [];

    // Group by quantile
    const quantileMap = new Map<number, Record<string, string | number | null>>();

    quantileLossData.forEach(d => {
      const quantile = Number(d.quantile);
      if (!quantileMap.has(quantile)) {
        quantileMap.set(quantile, { quantile: quantile.toFixed(2) });
      }
      const entry = quantileMap.get(quantile)!;
      entry[d.method] = d.metric_value;
    });

    return Array.from(quantileMap.values()).sort(
      (a, b) => parseFloat(a.quantile as string) - parseFloat(b.quantile as string)
    );
  }, [quantileLossData]);

  // Transform log loss data for bar chart
  const logLossChartData = useMemo(() => {
    if (logLossData.length === 0) return [];

    // Average log loss per method
    const methodMap = new Map<string, { sum: number; count: number }>();

    logLossData.forEach(d => {
      if (d.metric_value !== null) {
        if (!methodMap.has(d.method)) {
          methodMap.set(d.method, { sum: 0, count: 0 });
        }
        const entry = methodMap.get(d.method)!;
        entry.sum += d.metric_value;
        entry.count += 1;
      }
    });

    return Array.from(methodMap.entries()).map(([method, { sum, count }]) => ({
      method,
      value: sum / count,
    }));
  }, [logLossData]);

  // Determine best performing model
  const bestModel = useMemo(() => {
    if (methods.length === 0) return null;

    // Calculate average quantile loss per method (test only)
    const quantileLossAvg = new Map<string, number>();
    // Count unique variables per method for quantile loss
    const quantileVarCounts = new Map<string, Set<string>>();

    if (quantileLossData.length > 0) {
      const methodSums = new Map<string, { sum: number; count: number }>();
      quantileLossData.forEach(d => {
        if (d.metric_value !== null) {
          if (!methodSums.has(d.method)) {
            methodSums.set(d.method, { sum: 0, count: 0 });
          }
          const entry = methodSums.get(d.method)!;
          entry.sum += d.metric_value;
          entry.count += 1;

          // Track unique variables
          if (!quantileVarCounts.has(d.method)) {
            quantileVarCounts.set(d.method, new Set());
          }
          quantileVarCounts.get(d.method)!.add(d.variable);
        }
      });
      methodSums.forEach((value, method) => {
        quantileLossAvg.set(method, value.sum / value.count);
      });
    }

    // Calculate average log loss per method (test only, already have this in logLossChartData)
    const logLossAvg = new Map<string, number>();
    // Count unique variables per method for log loss
    const logLossVarCounts = new Map<string, Set<string>>();

    logLossData.forEach(d => {
      if (d.metric_value !== null) {
        if (!logLossVarCounts.has(d.method)) {
          logLossVarCounts.set(d.method, new Set());
        }
        logLossVarCounts.get(d.method)!.add(d.variable);
      }
    });

    logLossChartData.forEach(({ method, value }) => {
      logLossAvg.set(method, value);
    });

    // Rank methods by each metric (lower is better, so rank 1 is best)
    const rankMethods = (avgMap: Map<string, number>): Map<string, number> => {
      const sorted = Array.from(avgMap.entries()).sort((a, b) => a[1] - b[1]);
      const ranks = new Map<string, number>();
      sorted.forEach(([method], index) => {
        ranks.set(method, index + 1);
      });
      return ranks;
    };

    const quantileRanks = rankMethods(quantileLossAvg);
    const logLossRanks = rankMethods(logLossAvg);

    // Calculate weighted combined rank (weighted by number of variables of each type)
    // This matches autoimpute's select_best_model_dual_metrics approach
    const combinedRanks = new Map<string, number>();
    methods.forEach(method => {
      const qRank = quantileRanks.get(method);
      const lRank = logLossRanks.get(method);
      const nQuantileVars = quantileVarCounts.get(method)?.size || 0;
      const nLogLossVars = logLossVarCounts.get(method)?.size || 0;
      const totalVars = nQuantileVars + nLogLossVars;

      if (totalVars > 0) {
        let weightedRank = 0;
        if (qRank !== undefined) {
          weightedRank += nQuantileVars * qRank;
        }
        if (lRank !== undefined) {
          weightedRank += nLogLossVars * lRank;
        }
        combinedRanks.set(method, weightedRank / totalVars);
      } else {
        combinedRanks.set(method, Infinity);
      }
    });

    // Find best method (lowest combined rank)
    let bestMethod = '';
    let bestRank = Infinity;
    combinedRanks.forEach((rank, method) => {
      if (rank < bestRank) {
        bestRank = rank;
        bestMethod = method;
      }
    });

    // Calculate train/test ratios for the best method
    let quantileTrainTestRatio: number | undefined;
    let logLossTrainTestRatio: number | undefined;

    // Quantile loss train/test ratio
    const bestQuantileTrain = benchmarkData.filter(
      d => d.method === bestMethod && d.metric_name === 'quantile_loss' && d.split === 'train' && d.metric_value !== null
    );
    const bestQuantileTest = benchmarkData.filter(
      d => d.method === bestMethod && d.metric_name === 'quantile_loss' && d.split === 'test' && d.metric_value !== null
    );

    if (bestQuantileTrain.length > 0 && bestQuantileTest.length > 0) {
      const trainAvg = bestQuantileTrain.reduce((sum, d) => sum + d.metric_value!, 0) / bestQuantileTrain.length;
      const testAvg = bestQuantileTest.reduce((sum, d) => sum + d.metric_value!, 0) / bestQuantileTest.length;
      quantileTrainTestRatio = testAvg / trainAvg;
    }

    // Log loss train/test ratio
    const bestLogLossTrain = benchmarkData.filter(
      d => d.method === bestMethod && d.metric_name === 'log_loss' && d.split === 'train' && d.metric_value !== null
    );
    const bestLogLossTest = benchmarkData.filter(
      d => d.method === bestMethod && d.metric_name === 'log_loss' && d.split === 'test' && d.metric_value !== null
    );

    if (bestLogLossTrain.length > 0 && bestLogLossTest.length > 0) {
      const trainAvg = bestLogLossTrain.reduce((sum, d) => sum + d.metric_value!, 0) / bestLogLossTrain.length;
      const testAvg = bestLogLossTest.reduce((sum, d) => sum + d.metric_value!, 0) / bestLogLossTest.length;
      logLossTrainTestRatio = testAvg / trainAvg;
    }

    return {
      method: bestMethod,
      quantileLoss: quantileLossAvg.get(bestMethod),
      logLoss: logLossAvg.get(bestMethod),
      quantileTrainTestRatio,
      logLossTrainTestRatio,
    };
  }, [methods, quantileLossData, logLossChartData, benchmarkData]);

  // Set default selected method to best model
  useMemo(() => {
    if (bestModel && bestModel.method && !selectedMethod) {
      setSelectedMethod(bestModel.method);
    }
  }, [bestModel, selectedMethod]);

  // Prepare train/test comparison data for selected method
  const trainTestData = useMemo(() => {
    if (!selectedMethod) return { quantile: [], logLoss: [] };

    // Quantile loss train vs test
    const quantileTrainTest: Array<{ quantile: string; train: number | null; test: number | null }> = [];
    const quantileData = benchmarkData.filter(
      d => d.method === selectedMethod && d.metric_name === 'quantile_loss'
    );

    if (quantileData.length > 0) {
      const quantileMap = new Map<string, { train: number | null; test: number | null }>();

      quantileData.forEach(d => {
        const q = typeof d.quantile === 'number' ? d.quantile.toFixed(2) : String(d.quantile || '');
        // Skip 'mean' quantiles
        if (q.toLowerCase().includes('mean')) return;

        if (!quantileMap.has(q)) {
          quantileMap.set(q, { train: null, test: null });
        }
        const entry = quantileMap.get(q)!;
        if (d.split === 'train') entry.train = d.metric_value;
        if (d.split === 'test') entry.test = d.metric_value;
      });

      quantileMap.forEach((value, quantile) => {
        quantileTrainTest.push({ quantile, ...value });
      });

      quantileTrainTest.sort((a, b) => parseFloat(a.quantile) - parseFloat(b.quantile));
    }

    // Log loss train vs test (average across variables)
    const logLossTrainTest: Array<{ category: string; train: number; test: number }> = [];
    const logData = benchmarkData.filter(
      d => d.method === selectedMethod && d.metric_name === 'log_loss' && d.metric_value !== null
    );

    if (logData.length > 0) {
      const trainVals: number[] = [];
      const testVals: number[] = [];

      logData.forEach(d => {
        if (d.split === 'train') trainVals.push(d.metric_value!);
        if (d.split === 'test') testVals.push(d.metric_value!);
      });

      if (trainVals.length > 0 || testVals.length > 0) {
        const trainAvg = trainVals.length > 0 ? trainVals.reduce((a, b) => a + b, 0) / trainVals.length : 0;
        const testAvg = testVals.length > 0 ? testVals.reduce((a, b) => a + b, 0) / testVals.length : 0;

        logLossTrainTest.push({
          category: 'Average',
          train: trainAvg,
          test: testAvg,
        });
      }
    }

    return {
      quantile: quantileTrainTest,
      logLoss: logLossTrainTest,
    };
  }, [selectedMethod, benchmarkData]);

  const hasQuantileTrainTest = trainTestData.quantile.length > 0;
  const hasLogLossTrainTest = trainTestData.logLoss.length > 0;

  // Filter methods that have train/test data
  const methodsWithData = useMemo(() => {
    const validMethods = new Set<string>();

    methods.forEach(method => {
      const methodQuantileData = benchmarkData.filter(
        d => d.method === method && d.metric_name === 'quantile_loss' && d.metric_value !== null
      );
      const methodLogLossData = benchmarkData.filter(
        d => d.method === method && d.metric_name === 'log_loss' && d.metric_value !== null
      );

      if (methodQuantileData.length > 0 || methodLogLossData.length > 0) {
        validMethods.add(method);
      }
    });

    return Array.from(validMethods);
  }, [methods, benchmarkData]);

  const methodsWithoutData = methods.filter(m => !methodsWithData.includes(m));

  if (!hasBenchmarkData) {
    return null;
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6 text-gray-900">
        Benchmarking imputation methods
      </h2>

      {/* Best Model Highlight */}
      {bestModel && bestModel.method && (
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex-shrink-0 w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900">
                Best performing model: <span className="text-green-700">{bestModel.method}</span>
              </h3>
              <p className="text-sm text-gray-600">Based on combined performance across all metrics</p>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            {bestModel.quantileLoss !== undefined && (
              <div className="flex flex-col gap-1">
                <div>
                  <span className="text-gray-700">Avg. quantile loss (test): </span>
                  <span className="font-semibold text-gray-900">{bestModel.quantileLoss.toFixed(6)}</span>
                </div>
                {bestModel.quantileTrainTestRatio !== undefined && (
                  <span className={`text-xs ${bestModel.quantileTrainTestRatio > 1.1 ? 'text-amber-600' : 'text-gray-700'}`}>
                    Train/test ratio: {bestModel.quantileTrainTestRatio.toFixed(3)}
                  </span>
                )}
              </div>
            )}
            {bestModel.logLoss !== undefined && (
              <div className="flex flex-col gap-1">
                <div>
                  <span className="text-gray-700">Avg. log loss (test): </span>
                  <span className="font-semibold text-gray-900">{bestModel.logLoss.toFixed(6)}</span>
                </div>
                {bestModel.logLossTrainTestRatio !== undefined && (
                  <span className={`text-xs ${bestModel.logLossTrainTestRatio > 1.1 ? 'text-amber-600' : 'text-gray-700'}`}>
                    Train/test ratio: {bestModel.logLossTrainTestRatio.toFixed(3)}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Note about methods without data */}
      {methodsWithoutData.length > 0 && (
        <div className="mb-6 p-3 bg-gray-50 border border-gray-300 rounded-md">
          <p className="text-xs text-gray-600">
            <strong>Note:</strong> {methodsWithoutData.length === 1 ? 'The following method does' : 'The following methods do'} not appear in visualizations because {methodsWithoutData.length === 1 ? 'it does' : 'they do'} not support imputation of the selected variables due to variable types: <span className="font-mono">{methodsWithoutData.join(', ')}</span>
          </p>
        </div>
      )}

      <div className="space-y-8">
        {/* Quantile Loss Comparison */}
        {quantileChartData.length > 0 && (
          <div>
            <h3 className="text-xl font-semibold mb-4 text-gray-900">
              Test quantile loss across quantiles for different imputation methods
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={quantileChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="quantile"
                  label={{ value: 'Quantiles', position: 'insideBottom', offset: -5 }}
                  tick={{ fill: '#666' }}
                />
                <YAxis
                  label={{ value: 'Test quantile loss', angle: -90, position: 'insideLeft' }}
                  tick={{ fill: '#666' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    color: '#000',
                  }}
                  labelStyle={{ color: '#000', fontWeight: 'bold' }}
                  itemStyle={{ color: '#000' }}
                  formatter={(value: number) => value.toFixed(6)}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                {methods.map((method, index) => (
                  <Bar
                    key={method}
                    dataKey={method}
                    fill={getMethodColor(method, index)}
                    name={method}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-gray-700">
                <strong>Quantile loss</strong> measures how well the imputation method predicts different quantiles of the distribution for numerical variables, creating an asymmetric loss function that penalizes under-prediction more heavily for higher quantiles and over-prediction more heavily for lower quantiles.
                <br />
                Lower values indicate better performance.
              </p>
            </div>
          </div>
        )}

        {/* Log Loss Comparison */}
        {logLossChartData.length > 0 && (
          <div>
            <h3 className="text-xl font-semibold mb-4 text-gray-900">
              Test log loss across different imputation methods
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={logLossChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="method"
                  tick={{ fill: '#666' }}
                />
                <YAxis
                  label={{ value: 'Log loss', angle: -90, position: 'insideLeft' }}
                  tick={{ fill: '#666' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    color: '#000',
                  }}
                  labelStyle={{ color: '#000', fontWeight: 'bold' }}
                  itemStyle={{ color: '#000' }}
                  formatter={(value: number) => [value.toFixed(6), 'Log loss']}
                />
                <Bar dataKey="value">
                  {logLossChartData.map((entry, index) => (
                    <Cell key={entry.method} fill={getMethodColor(entry.method, index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-gray-700">
                <strong>Log loss</strong> measures how well the imputation method predicts categorical and boolean variables by evaluating the accuracy of predicted probabilities. It heavily penalizes confident misclassifications, such that a perfect classifier would have a log loss of 0, while worse predictions yield increasingly higher values.
              </p>
            </div>
          </div>
        )}

        {/* Train/Test Overfitting Assessment */}
        {(hasQuantileTrainTest || hasLogLossTrainTest) && methods.length > 0 && (
          <div className="mt-8 pt-8 border-t-2 border-gray-200">
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-2 text-gray-900">
                Train vs test performance
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                Compare training and test set performance to assess potential overfitting or underfitting.
              </p>

              {/* Method Selector */}
              <div className="flex items-center gap-3">
                <label htmlFor="method-select" className="text-sm font-medium text-gray-700">
                  Select method:
                </label>
                <select
                  id="method-select"
                  value={selectedMethod}
                  onChange={(e) => setSelectedMethod(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
                >
                  {methodsWithData.map((method) => (
                    <option key={method} value={method}>
                      {method} {bestModel && method === bestModel.method ? '' : ''}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className={`grid gap-6 ${hasQuantileTrainTest && hasLogLossTrainTest ? 'grid-cols-2' : 'grid-cols-1'}`}>
              {/* Quantile Loss Train/Test */}
              {hasQuantileTrainTest && (
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-gray-900">Quantile loss: train vs test</h4>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={trainTestData.quantile}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis
                        dataKey="quantile"
                        label={{ value: 'Quantiles', position: 'insideBottom', offset: -5 }}
                        tick={{ fill: '#666' }}
                      />
                      <YAxis
                        label={{ value: 'Quantile loss', angle: -90, position: 'insideLeft' }}
                        tick={{ fill: '#666' }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #ccc',
                          color: '#000',
                        }}
                        labelStyle={{ color: '#000', fontWeight: 'bold' }}
                        itemStyle={{ color: '#000' }}
                        formatter={(value: number) => value.toFixed(6)}
                      />
                      <Legend wrapperStyle={{ paddingTop: '25px' }} />
                      <Bar dataKey="train" fill="#06b6d4" name="Train" />
                      <Bar dataKey="test" fill="#16a34a" name="Test" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Log Loss Train/Test */}
              {hasLogLossTrainTest && (
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-gray-900">Log loss: train vs test</h4>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={trainTestData.logLoss}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis
                        dataKey="category"
                        tick={{ fill: '#666' }}
                      />
                      <YAxis
                        label={{ value: 'Log loss', angle: -90, position: 'insideLeft' }}
                        tick={{ fill: '#666' }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #ccc',
                          color: '#000',
                        }}
                        labelStyle={{ color: '#000', fontWeight: 'bold' }}
                        itemStyle={{ color: '#000' }}
                        formatter={(value: number) => value.toFixed(6)}
                      />
                      <Legend wrapperStyle={{ paddingTop: '25px' }} />
                      <Bar dataKey="train" fill="#06b6d4" name="Train" />
                      <Bar dataKey="test" fill="#16a34a" name="Test" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-gray-700">
                <strong>Overfitting assessment:</strong> When test performance (green bars) is significantly worse than train performance (cyan bars), it suggests the model may be overfitting to the training data and not generalizing well to unseen data. If both train and test performances are poor, the model may be underfitting and failing to capture underlying patterns.
                <br />
                Healthy performance is indicated by similar train and test metrics, with both being reasonably low.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
