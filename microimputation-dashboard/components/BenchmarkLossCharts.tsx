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
  ReferenceLine,
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
    const quantileMap = new Map<number, Record<string, any>>();

    quantileLossData.forEach(d => {
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

  if (!hasBenchmarkData) {
    return null;
  }

  return (
    <div className="space-y-8">
      {/* Quantile Loss Comparison */}
      {quantileChartData.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Test Quantile Loss Across Quantiles for Different Imputation Methods
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
                label={{ value: 'Test Quantile Loss', angle: -90, position: 'insideLeft' }}
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

      {/* Log Loss Comparison */}
      {logLossChartData.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4 text-gray-900">
            Log Loss Comparison Across Methods
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={logLossChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="method"
                tick={{ fill: '#666' }}
              />
              <YAxis
                label={{ value: 'Log Loss', angle: -90, position: 'insideLeft' }}
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
