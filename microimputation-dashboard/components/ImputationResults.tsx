'use client';

import { useMemo } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface ImputationResultsProps {
  data: ImputationDataPoint[];
}

interface DistributionMetric {
  variable: string;
  method: string;
  metricName: string;
  value: number;
}

export default function ImputationResults({ data }: ImputationResultsProps) {
  // Filter for distribution distance data
  const distributionData = useMemo(() => {
    return data.filter(d => d.type === 'distribution_distance');
  }, [data]);

  // Group by metric type
  const { wassersteinData, klDivergenceData } = useMemo(() => {
    const wasserstein: DistributionMetric[] = [];
    const klDiv: DistributionMetric[] = [];

    distributionData.forEach(d => {
      const metric: DistributionMetric = {
        variable: d.variable,
        method: d.method,
        metricName: d.metric_name,
        value: d.metric_value ?? 0,
      };

      if (d.metric_name === 'wasserstein_distance') {
        wasserstein.push(metric);
      } else if (d.metric_name === 'kl_divergence') {
        klDiv.push(metric);
      }
    });

    // Sort by value (ascending - lower is better)
    wasserstein.sort((a, b) => a.value - b.value);
    klDiv.sort((a, b) => a.value - b.value);

    return {
      wassersteinData: wasserstein,
      klDivergenceData: klDiv
    };
  }, [distributionData]);

  const hasWasserstein = wassersteinData.length > 0;
  const hasKLDivergence = klDivergenceData.length > 0;

  if (!hasWasserstein && !hasKLDivergence) {
    return null;
  }

  // Color function based on value quality (lower is better)
  const getWassersteinColor = (value: number): string => {
    if (value < 0.01) return '#16a34a'; // Dark green - excellent
    if (value < 0.05) return '#22c55e'; // Green - good
    if (value < 0.1) return '#eab308'; // Yellow - moderate
    if (value < 0.2) return '#f97316'; // Orange - fair
    return '#ef4444'; // Red - poor
  };

  const getKLColor = (value: number): string => {
    if (value < 0.1) return '#16a34a'; // Dark green - excellent
    if (value < 0.5) return '#22c55e'; // Green - good
    if (value < 1.0) return '#eab308'; // Yellow - moderate
    if (value < 5.0) return '#f97316'; // Orange - fair
    return '#ef4444'; // Red - poor
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900">
          Imputation results
        </h2>
        <p className="text-sm text-gray-600">
          Distributional quality metrics comparing imputed values to true values
        </p>
      </div>

      {/* Wasserstein Distance Section */}
      {hasWasserstein && (
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Numerical variables (Wasserstein distance)
          </h3>

          {/* Explanation */}
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm text-gray-700 mb-2">
              <strong>What is Wasserstein distance?</strong> Also known as "Earth Mover's Distance,"
              this metric measures how much "work" is needed to transform one probability distribution
              into another. Think of it as the minimum cost to rearrange one pile of dirt to match
              another pile's shape.
            </p>
            <p className="text-sm text-gray-700 mb-2">
              <strong>Why use it for imputation?</strong> Wasserstein distance is ideal for numerical
              variables because it considers the actual distances between values, not just whether
              they match exactly. A value of 0 means perfect imputation, and larger values indicate
              greater differences between imputed and true distributions.
            </p>
            <p className="text-sm text-gray-700">
              <strong>Interpretation:</strong> Values closer to 0 are better. Generally, values below
              0.05 indicate good imputation quality, while values above 0.2 suggest significant
              distributional differences.
            </p>
          </div>

          {/* Bar chart */}
          <div className="mb-4">
            <ResponsiveContainer width="100%" height={Math.max(200, wassersteinData.length * 60)}>
              <BarChart
                data={wassersteinData}
                layout="vertical"
                margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fill: '#000000' }} />
                <YAxis type="category" dataKey="variable" width={90} tick={{ fill: '#000000' }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(6), 'Wasserstein Distance']}
                />
                <Legend wrapperStyle={{ color: '#000000' }} />
                <Bar dataKey="value" name="Wasserstein Distance">
                  {wassersteinData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={getWassersteinColor(entry.value)}
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
                    Variable
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Wasserstein Distance
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quality Assessment
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {wassersteinData.map((item) => {
                  let assessment = '';
                  let assessmentColor = '';

                  if (item.value < 0.01) {
                    assessment = 'Excellent';
                    assessmentColor = 'text-green-700 font-semibold';
                  } else if (item.value < 0.05) {
                    assessment = 'Good';
                    assessmentColor = 'text-green-600';
                  } else if (item.value < 0.1) {
                    assessment = 'Moderate';
                    assessmentColor = 'text-yellow-600';
                  } else if (item.value < 0.2) {
                    assessment = 'Fair';
                    assessmentColor = 'text-orange-600';
                  } else {
                    assessment = 'Poor';
                    assessmentColor = 'text-red-600 font-semibold';
                  }

                  return (
                    <tr key={item.variable}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono font-medium text-gray-900">
                        {item.variable}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                        {item.value.toFixed(6)}
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

      {/* KL Divergence Section */}
      {hasKLDivergence && (
        <div className={hasWasserstein ? 'mt-8 pt-8 border-t-2 border-gray-200' : ''}>
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Categorical variables (KL-divergence)
          </h3>

          {/* Explanation */}
          <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded-md">
            <p className="text-sm text-gray-700 mb-2">
              <strong>What is KL-divergence?</strong> Kullback-Leibler divergence measures how much
              one probability distribution differs from another. It quantifies the "information lost"
              when using the imputed distribution to approximate the true distribution.
            </p>
            <p className="text-sm text-gray-700 mb-2">
              <strong>Why use it for categorical variables?</strong> KL-divergence is particularly
              useful for categorical data because it compares probability distributions across
              categories. It's sensitive to differences in how probabilities are distributed across
              all possible categories.
            </p>
            <p className="text-sm text-gray-700">
              <strong>Interpretation:</strong> A value of 0 means perfect match. Values below 0.5
              indicate good imputation, while values above 5.0 suggest substantial distributional
              differences. Note that KL-divergence is not symmetric and can range from 0 to infinity.
            </p>
          </div>

          {/* Bar chart */}
          <div className="mb-4">
            <ResponsiveContainer width="100%" height={Math.max(200, klDivergenceData.length * 60)}>
              <BarChart
                data={klDivergenceData}
                layout="vertical"
                margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fill: '#000000' }} />
                <YAxis type="category" dataKey="variable" width={90} tick={{ fill: '#000000' }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(6), 'KL-Divergence']}
                />
                <Legend wrapperStyle={{ color: '#000000' }} />
                <Bar dataKey="value" name="KL-Divergence">
                  {klDivergenceData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={getKLColor(entry.value)}
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
                    Variable
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    KL-Divergence
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quality Assessment
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {klDivergenceData.map((item) => {
                  let assessment = '';
                  let assessmentColor = '';

                  if (item.value < 0.1) {
                    assessment = 'Excellent';
                    assessmentColor = 'text-green-700 font-semibold';
                  } else if (item.value < 0.5) {
                    assessment = 'Good';
                    assessmentColor = 'text-green-600';
                  } else if (item.value < 1.0) {
                    assessment = 'Moderate';
                    assessmentColor = 'text-yellow-600';
                  } else if (item.value < 5.0) {
                    assessment = 'Fair';
                    assessmentColor = 'text-orange-600';
                  } else {
                    assessment = 'Poor';
                    assessmentColor = 'text-red-600 font-semibold';
                  }

                  return (
                    <tr key={item.variable}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono font-medium text-gray-900">
                        {item.variable}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                        {item.value.toFixed(6)}
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
