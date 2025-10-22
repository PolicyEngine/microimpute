'use client';

import { useMemo } from 'react';
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

interface PerVariableChartsProps {
  data: ImputationDataPoint[];
  variable: string;
  metricType: 'quantile_loss' | 'log_loss';
}

export default function PerVariableCharts({
  data,
  variable,
  metricType,
}: PerVariableChartsProps) {
  // Filter data for this specific variable
  const variableData = useMemo(() => {
    return data.filter(
      (d) =>
        d.type === 'benchmark_loss' &&
        d.variable === variable &&
        d.metric_name === metricType &&
        d.split === 'test'
    );
  }, [data, variable, metricType]);

  const methods = useMemo(() => {
    return Array.from(new Set(variableData.map((d) => d.method)));
  }, [variableData]);

  // For numerical variables (quantile_loss), show quantile breakdown
  const quantileChartData = useMemo(() => {
    if (metricType !== 'quantile_loss') return [];

    const numericData = variableData.filter(
      (d) =>
        typeof d.quantile === 'number' && d.quantile >= 0 && d.quantile <= 1
    );

    const quantileMap = new Map<number, Record<string, any>>();

    numericData.forEach((d) => {
      const quantile = Number(d.quantile);
      if (!quantileMap.has(quantile)) {
        quantileMap.set(quantile, { quantile: quantile.toFixed(2) });
      }
      const entry = quantileMap.get(quantile)!;
      entry[d.method] = d.metric_value;
    });

    return Array.from(quantileMap.values()).sort(
      (a, b) => parseFloat(a.quantile) - parseFloat(b.quantile)
    );
  }, [variableData, metricType]);

  // For categorical variables (log_loss), show simple bar comparison
  const logLossChartData = useMemo(() => {
    if (metricType !== 'log_loss') return [];

    const methodMap = new Map<string, { sum: number; count: number }>();

    variableData.forEach((d) => {
      if (d.metric_value !== null) {
        if (!methodMap.has(d.method)) {
          methodMap.set(d.method, { sum: 0, count: 0 });
        }
        const entry = methodMap.get(d.method)!;
        entry.sum += d.metric_value;
        entry.count += 1;
      }
    });

    return Array.from(methodMap.entries()).map(
      ([method, { sum, count }]) => ({
        method,
        value: sum / count,
      })
    );
  }, [variableData, metricType]);

  if (variableData.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No data available for variable: {variable}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {metricType === 'quantile_loss' && quantileChartData.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-700">
            Quantile Loss by Method for "{variable}"
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={quantileChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="quantile"
                label={{
                  value: 'Quantiles',
                  position: 'insideBottom',
                  offset: -5,
                }}
                tick={{ fill: '#666' }}
              />
              <YAxis
                label={{
                  value: 'Test Quantile Loss',
                  angle: -90,
                  position: 'insideLeft',
                }}
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
        </div>
      )}

      {metricType === 'log_loss' && logLossChartData.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-700">
            Log Loss by Method for "{variable}"
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={logLossChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="method" tick={{ fill: '#666' }} />
              <YAxis
                label={{
                  value: 'Log Loss',
                  angle: -90,
                  position: 'insideLeft',
                }}
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
                formatter={(value: number) => [value.toFixed(6), 'Log Loss']}
              />
              <Bar dataKey="value">
                {logLossChartData.map((entry, index) => (
                  <Cell key={entry.method} fill={getMethodColor(entry.method, index)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
