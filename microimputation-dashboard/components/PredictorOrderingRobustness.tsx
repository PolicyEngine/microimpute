'use client';

import { useMemo } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface PredictorOrderingRobustnessProps {
  data: ImputationDataPoint[];
}

interface ProgressiveStep {
  step: number;
  predictorAdded: string;
  predictors: string[];
  cumulativeImprovement: number;
  marginalImprovement: number;
}

interface PredictorImportance {
  predictor: string;
  relativeImpact: number;
  lossIncrease: number;
}

export default function PredictorOrderingRobustness({ data }: PredictorOrderingRobustnessProps) {
  // Filter for progressive inclusion data
  const progressiveInclusionData = useMemo(() => {
    return data.filter(d => d.type === 'progressive_inclusion');
  }, [data]);

  // Filter for predictor importance data
  const predictorImportanceData = useMemo(() => {
    return data.filter(d => d.type === 'predictor_importance');
  }, [data]);

  // Parse progressive inclusion steps
  const progressiveSteps = useMemo(() => {
    const stepData: ProgressiveStep[] = [];
    const cumulativeData = progressiveInclusionData.filter(
      d => d.metric_name === 'cumulative_improvement'
    );

    cumulativeData.forEach(d => {
      try {
        const additionalInfo = typeof d.additional_info === 'string'
          ? JSON.parse(d.additional_info)
          : d.additional_info;

        const step = additionalInfo?.step;
        const predictorAdded = additionalInfo?.predictor_added;
        const predictors = additionalInfo?.predictors || [];

        if (step !== undefined && predictorAdded) {
          // Find corresponding marginal improvement
          const marginalData = progressiveInclusionData.find(
            m => m.metric_name === 'marginal_improvement' &&
                 JSON.parse(typeof m.additional_info === 'string' ? m.additional_info : JSON.stringify(m.additional_info))?.step === step
          );

          stepData.push({
            step,
            predictorAdded,
            predictors,
            cumulativeImprovement: d.metric_value ?? 0,
            marginalImprovement: marginalData?.metric_value ?? 0,
          });
        }
      } catch (e) {
        console.error('Failed to parse progressive inclusion data:', e);
      }
    });

    return stepData.sort((a, b) => a.step - b.step);
  }, [progressiveInclusionData]);

  // Parse predictor importance
  const importanceData = useMemo(() => {
    const importanceMap = new Map<string, PredictorImportance>();

    predictorImportanceData.forEach(d => {
      try {
        const additionalInfo = typeof d.additional_info === 'string'
          ? JSON.parse(d.additional_info)
          : d.additional_info;

        const predictor = additionalInfo?.removed_predictor || d.variable;

        if (predictor) {
          if (!importanceMap.has(predictor)) {
            importanceMap.set(predictor, {
              predictor,
              relativeImpact: 0,
              lossIncrease: 0,
            });
          }

          const entry = importanceMap.get(predictor)!;
          if (d.metric_name === 'relative_impact') {
            entry.relativeImpact = d.metric_value ?? 0;
          } else if (d.metric_name === 'loss_increase') {
            entry.lossIncrease = d.metric_value ?? 0;
          }
        }
      } catch (e) {
        console.error('Failed to parse predictor importance data:', e);
      }
    });

    return Array.from(importanceMap.values()).sort(
      (a, b) => Math.abs(b.relativeImpact) - Math.abs(a.relativeImpact)
    );
  }, [predictorImportanceData]);

  const hasProgressiveData = progressiveSteps.length > 0;
  const hasImportanceData = importanceData.length > 0;

  if (!hasProgressiveData && !hasImportanceData) {
    return null;
  }

  // Find best combination (highest cumulative improvement)
  const bestCombination = progressiveSteps.reduce((best, current) =>
    current.cumulativeImprovement > best.cumulativeImprovement ? current : best,
    progressiveSteps[0]
  );

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900">
          Predictor selection and robustness
        </h2>
        <p className="text-sm text-gray-600">
          Analysis of predictor combinations and their impact on model performance
        </p>
      </div>

      {/* Progressive Inclusion Section */}
      {hasProgressiveData && (
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Predictor addition order
          </h3>

          {/* Explanation */}
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm text-gray-700 mb-2">
              <strong>How this works:</strong> This analysis adds predictors one at a time,
              choosing the predictor that improves performance the most at each step. This
              step-by-step approach is efficient but doesn't test
              every possible combination of predictors.
            </p>
            <p className="text-sm text-gray-700">
              <strong>Reading the chart:</strong> The bars show cumulative improvement from
              baseline as predictors are added. Larger improvements indicate more valuable
              predictor combinations.
            </p>
          </div>

          {/* Best Combination Highlight */}
          {bestCombination && (
            <div className="mb-6 p-4 bg-green-50 border-2 border-green-500 rounded-md">
              <h4 className="text-md font-semibold text-gray-900 mb-2">
                Best predictor combination
              </h4>
              <div className="flex items-start gap-4">
                <div className="flex-1">
                  <p className="text-sm text-gray-700 mb-1">
                    <strong>Predictors:</strong>{' '}
                    <span className="font-mono text-gray-900">
                      {bestCombination.predictors.join(' → ')}
                    </span>
                  </p>
                  <p className="text-sm text-gray-700">
                    <strong>Cumulative improvement:</strong>{' '}
                    <span className="font-semibold text-green-700">
                      {(bestCombination.cumulativeImprovement * 100).toFixed(3)}%
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Step-by-step visualization */}
          <div className="space-y-4">
            {progressiveSteps.map((step) => {
              const isPositive = step.marginalImprovement >= 0;
              const isBest = step.step === bestCombination?.step;

              return (
                <div
                  key={step.step}
                  className={`p-4 rounded-md border-2 ${
                    isBest
                      ? 'bg-green-50 border-green-500'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-white ${
                        isBest ? 'bg-green-600' : 'bg-blue-600'
                      }`}>
                        {step.step}
                      </div>
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-gray-600">Add:</span>
                        <span className="font-mono font-semibold text-gray-900">
                          {step.predictorAdded}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                        <div>
                          <span className="text-xs text-gray-600">Marginal improvement:</span>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  isPositive ? 'bg-green-500' : 'bg-red-500'
                                }`}
                                style={{
                                  width: `${Math.min(Math.abs(step.marginalImprovement) * 5000, 100)}%`,
                                }}
                              />
                            </div>
                            <span className={`text-sm font-semibold ${
                              isPositive ? 'text-green-700' : 'text-red-700'
                            }`}>
                              {isPositive ? '+' : ''}{(step.marginalImprovement * 100).toFixed(3)}%
                            </span>
                          </div>
                        </div>

                        <div>
                          <span className="text-xs text-gray-600">Cumulative improvement:</span>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-gray-200 rounded-full h-2">
                              <div
                                className="h-2 rounded-full bg-blue-500"
                                style={{
                                  width: `${Math.min(Math.abs(step.cumulativeImprovement) * 5000, 100)}%`,
                                }}
                              />
                            </div>
                            <span className="text-sm font-semibold text-blue-700">
                              {(step.cumulativeImprovement * 100).toFixed(3)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="mt-2 text-xs text-gray-500">
                        Current predictors: {step.predictors.join(' → ')}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Predictor Importance Section */}
      {hasImportanceData && (
        <div className="mt-8 pt-8 border-t-2 border-gray-200">
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Predictor robustness check
          </h3>

          {/* Explanation */}
          <div className="mb-6 p-4 bg-blue-50 border border-purple-200 rounded-md">
            <p className="text-sm text-gray-700 mb-2">
              <strong>What this shows:</strong> This analysis measures how much performance
              degrades when each predictor is removed. Predictors that cause large performance
              drops when removed are critical to the model's accuracy.
            </p>
            <p className="text-sm text-gray-700">
              <strong>Reading the chart:</strong> Positive values (bars pointing right) indicate
              performance worsens when the predictor is removed, meaning the predictor is helpful.
              Negative values suggest removing the predictor might actually improve performance.
            </p>
          </div>

          {/* Bar chart */}
          <div className="mb-4">
            <ResponsiveContainer width="100%" height={Math.max(300, importanceData.length * 50)}>
              <BarChart
                data={importanceData}
                layout="vertical"
                margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={(val) => `${val.toFixed(1)}%`} tick={{ fill: '#000000' }} />
                <YAxis type="category" dataKey="predictor" width={90} tick={{ fill: '#000000' }} />
                <Tooltip
                  formatter={(value: number, name: string) => {
                    if (name === 'relativeImpact') {
                      return [`${value.toFixed(3)}%`, 'Relative Impact'];
                    }
                    return [value.toFixed(6), 'Loss Increase'];
                  }}
                />
                <Legend wrapperStyle={{ color: '#000000' }} />
                <Bar dataKey="relativeImpact" name="Relative Impact (%)">
                  {importanceData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.relativeImpact >= 0 ? '#ef4444' : '#22c55e'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed table */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predictor
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Impact when removed
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Loss increase
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Assessment
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {importanceData.map((item) => {
                  const isHelpful = item.relativeImpact > 1;
                  const isCritical = item.relativeImpact > 10;
                  const isHarmful = item.relativeImpact < -1;

                  let assessment = 'Minimal impact';
                  let assessmentColor = 'text-gray-600';

                  if (isCritical) {
                    assessment = 'Critical predictor';
                    assessmentColor = 'text-red-700 font-semibold';
                  } else if (isHelpful) {
                    assessment = 'Helpful predictor';
                    assessmentColor = 'text-orange-600';
                  } else if (isHarmful) {
                    assessment = 'Consider removing';
                    assessmentColor = 'text-green-600';
                  }

                  return (
                    <tr key={item.predictor}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono font-medium text-gray-900">
                        {item.predictor}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <span className={item.relativeImpact >= 0 ? 'text-red-600' : 'text-green-600'}>
                          {item.relativeImpact >= 0 ? '+' : ''}{item.relativeImpact.toFixed(3)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                        {item.lossIncrease >= 0 ? '+' : ''}{item.lossIncrease.toFixed(6)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm ${assessmentColor}`}>
                        {assessment}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
