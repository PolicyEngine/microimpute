'use client';

import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  ErrorBar,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { ImputationDataPoint } from '@/types/imputation';
import { getMethodColor, GRID_COLOR, LINE_COLOR } from '@/utils/colors';
import ChartLegend from './ChartLegend';

interface PerVariableChartsProps {
  data: ImputationDataPoint[];
  variable: string;
  metricType: 'quantile_loss' | 'log_loss';
}

const ERROR_BAR_STROKE = '#374151';

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
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

  // Use global method ordering from all benchmark data for consistent colors
  const allMethods = useMemo(() => {
    return Array.from(new Set(data.filter(d => d.type === 'benchmark_loss').map(d => d.method)));
  }, [data]);

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

    const quantileMap = new Map<number, Record<string, string | number | null | undefined>>();

    numericData.forEach((d) => {
      const quantile = Number(d.quantile);
      if (!quantileMap.has(quantile)) {
        quantileMap.set(quantile, { quantile: quantile.toFixed(2) });
      }
      const entry = quantileMap.get(quantile)!;
      entry[d.method] = d.metric_value;
      if (isFiniteNumber(d.metric_std)) {
        entry[`${d.method}__std`] = d.metric_std;
      }
    });

    return Array.from(quantileMap.values()).sort(
      (a, b) => parseFloat(a.quantile as string) - parseFloat(b.quantile as string)
    );
  }, [variableData, metricType]);

  const hasQuantileErrorBarsByMethod = useMemo(() => {
    const result = new Map<string, boolean>();
    methods.forEach((method) => {
      result.set(
        method,
        quantileChartData.some((row) => isFiniteNumber(row[`${method}__std`]))
      );
    });
    return result;
  }, [methods, quantileChartData]);

  // For categorical variables (log_loss), show simple bar comparison
  const logLossChartData = useMemo(() => {
    if (metricType !== 'log_loss') return [];

    const methodMap = new Map<string, { sum: number; count: number; stdSum: number; stdCount: number }>();

    variableData.forEach((d) => {
      if (d.metric_value !== null) {
        if (!methodMap.has(d.method)) {
          methodMap.set(d.method, { sum: 0, count: 0, stdSum: 0, stdCount: 0 });
        }
        const entry = methodMap.get(d.method)!;
        entry.sum += d.metric_value;
        entry.count += 1;
        if (isFiniteNumber(d.metric_std)) {
          entry.stdSum += d.metric_std;
          entry.stdCount += 1;
        }
      }
    });

    return Array.from(methodMap.entries()).map(
      ([method, { sum, count, stdSum, stdCount }]) => ({
        method,
        value: sum / count,
        std: stdCount > 0 ? stdSum / stdCount : undefined,
      })
    );
  }, [variableData, metricType]);

  const hasLogLossErrorBars = useMemo(() => {
    return logLossChartData.some((row) => isFiniteNumber(row.std));
  }, [logLossChartData]);

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
            Quantile Loss by Method for &quot;{variable}&quot;
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={quantileChartData}>
              <CartesianGrid stroke={GRID_COLOR} />
              <XAxis
                dataKey="quantile"
                label={{
                  value: 'Quantiles',
                  position: 'insideBottom',
                  offset: -5,
                }}
                tick={{ fill: '#333' }}
                axisLine={{ stroke: LINE_COLOR }}
                tickLine={{ stroke: LINE_COLOR }}
              />
              <YAxis
                width={100}
                label={{
                  value: 'Test Quantile Loss',
                  angle: -90,
                  position: 'center',
                }}
                tick={{ fill: '#333' }}
                axisLine={{ stroke: LINE_COLOR }}
                tickLine={{ stroke: LINE_COLOR }}
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
              <Legend content={<ChartLegend />} />
              {methods.map((method) => {
                const globalIndex = allMethods.indexOf(method);
                return (
                  <Bar
                    key={method}
                    dataKey={method}
                    fill={getMethodColor(method, globalIndex >= 0 ? globalIndex : 0)}
                    name={method}
                  >
                    {hasQuantileErrorBarsByMethod.get(method) && (
                      <ErrorBar
                        dataKey={`${method}__std`}
                        width={4}
                        stroke={ERROR_BAR_STROKE}
                      />
                    )}
                  </Bar>
                );
              })}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {metricType === 'log_loss' && logLossChartData.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-700">
            Log Loss by Method for &quot;{variable}&quot;
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={logLossChartData}>
              <CartesianGrid stroke={GRID_COLOR} />
              <XAxis dataKey="method" tick={{ fill: '#333' }}
                axisLine={{ stroke: LINE_COLOR }}
                tickLine={{ stroke: LINE_COLOR }} />
              <YAxis
                width={100}
                label={{
                  value: 'Log Loss',
                  angle: -90,
                  position: 'center',
                }}
                tick={{ fill: '#333' }}
                axisLine={{ stroke: LINE_COLOR }}
                tickLine={{ stroke: LINE_COLOR }}
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
                {hasLogLossErrorBars && (
                  <ErrorBar
                    dataKey="std"
                    width={4}
                    stroke={ERROR_BAR_STROKE}
                  />
                )}
                {logLossChartData.map((entry) => {
                  const globalIndex = allMethods.indexOf(entry.method);
                  return (
                    <Cell key={entry.method} fill={getMethodColor(entry.method, globalIndex >= 0 ? globalIndex : 0)} />
                  );
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
