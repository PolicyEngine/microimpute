'use client';

import { useMemo, useState } from 'react';
import { ImputationDataPoint } from '@/types/imputation';

interface PredictorCorrelationMatrixProps {
  data: ImputationDataPoint[];
}

interface CorrelationData {
  predictor1: string;
  predictor2: string;
  value: number;
}

export default function PredictorCorrelationMatrix({ data }: PredictorCorrelationMatrixProps) {
  // Filter for predictor_correlation data
  const correlationData = useMemo(() => {
    return data.filter(d => d.type === 'predictor_correlation');
  }, [data]);

  // Filter for predictor-target mutual information data
  const predictorTargetMIData = useMemo(() => {
    return data.filter(d => d.type === 'predictor_target_mi');
  }, [data]);

  // Check available correlation metrics
  const availableMetrics = useMemo(() => {
    const metrics = new Set(correlationData.map(d => d.metric_name));
    return Array.from(metrics);
  }, [correlationData]);

  // State for selected correlation metric
  const [selectedMetric, setSelectedMetric] = useState<string>('');

  // Set default metric to pearson if available, otherwise first available
  useMemo(() => {
    if (!selectedMetric && availableMetrics.length > 0) {
      setSelectedMetric(availableMetrics.includes('pearson') ? 'pearson' : availableMetrics[0]);
    }
  }, [availableMetrics, selectedMetric]);

  // Build correlation matrix data
  const { predictors, matrixData } = useMemo(() => {
    if (!selectedMetric) return { predictors: [], matrixData: new Map<string, Map<string, number>>() };

    // Filter data for selected metric
    const metricData = correlationData.filter(d => d.metric_name === selectedMetric);

    // Extract all unique predictors
    const predSet = new Set<string>();
    const correlations: CorrelationData[] = [];

    metricData.forEach(d => {
      const pred1 = d.variable;
      let pred2: string | undefined;

      try {
        const additionalInfo = typeof d.additional_info === 'string'
          ? JSON.parse(d.additional_info)
          : d.additional_info;
        pred2 = additionalInfo?.predictor2;
      } catch (e) {
        console.error('Failed to parse additional_info:', e);
      }

      if (pred1 && pred2) {
        predSet.add(pred1);
        predSet.add(pred2);
        correlations.push({
          predictor1: pred1,
          predictor2: pred2,
          value: d.metric_value ?? 0,
        });
      }
    });

    const predictorList = Array.from(predSet).sort();

    // Build symmetric matrix
    const matrix = new Map<string, Map<string, number>>();

    predictorList.forEach(p => {
      matrix.set(p, new Map<string, number>());
    });

    // Add diagonal (1.0 for self-correlation)
    predictorList.forEach(p => {
      matrix.get(p)!.set(p, 1.0);
    });

    // Add correlations (symmetric)
    correlations.forEach(({ predictor1, predictor2, value }) => {
      matrix.get(predictor1)!.set(predictor2, value);
      matrix.get(predictor2)!.set(predictor1, value);
    });

    return { predictors: predictorList, matrixData: matrix };
  }, [correlationData, selectedMetric]);

  // Build predictor-target mutual information matrix
  const { predictorsList, targetsList, miMatrixData } = useMemo(() => {
    if (predictorTargetMIData.length === 0) {
      return { predictorsList: [], targetsList: [], miMatrixData: new Map<string, Map<string, number>>() };
    }

    const predSet = new Set<string>();
    const targSet = new Set<string>();
    const miValues: Array<{ predictor: string; target: string; value: number }> = [];

    predictorTargetMIData.forEach(d => {
      const predictor = d.variable;
      let target: string | undefined;

      try {
        const additionalInfo = typeof d.additional_info === 'string'
          ? JSON.parse(d.additional_info)
          : d.additional_info;
        target = additionalInfo?.target;
      } catch (e) {
        console.error('Failed to parse additional_info:', e);
      }

      if (predictor && target && d.metric_value !== null) {
        predSet.add(predictor);
        targSet.add(target);
        miValues.push({
          predictor,
          target,
          value: d.metric_value,
        });
      }
    });

    const predList = Array.from(predSet).sort();
    const targList = Array.from(targSet).sort();

    // Build matrix
    const matrix = new Map<string, Map<string, number>>();
    predList.forEach(p => {
      matrix.set(p, new Map<string, number>());
    });

    miValues.forEach(({ predictor, target, value }) => {
      matrix.get(predictor)!.set(target, value);
    });

    return { predictorsList: predList, targetsList: targList, miMatrixData: matrix };
  }, [predictorTargetMIData]);

  const hasPredictorTargetMI = predictorsList.length > 0 && targetsList.length > 0;

  if (correlationData.length === 0 || predictors.length === 0) {
    return null;
  }

  // Helper function to get color based on correlation value
  const getColor = (value: number): string => {
    // Scale from -1 to 1
    // Negative: red shades, Positive: blue shades, Zero: white
    if (value === 1.0) return '#1e40af'; // Dark blue for diagonal
    if (value >= 0.7) return '#3b82f6'; // Blue
    if (value >= 0.4) return '#60a5fa'; // Light blue
    if (value >= 0.2) return '#93c5fd'; // Very light blue
    if (value >= -0.2) return '#f3f4f6'; // Nearly white
    if (value >= -0.4) return '#fca5a5'; // Light red
    if (value >= -0.7) return '#f87171'; // Red
    return '#ef4444'; // Dark red
  };

  // Helper function to get color based on mutual information value (0 to ~1)
  const getMIColor = (value: number): string => {
    // Scale from 0 (white) to high values (dark purple)
    if (value >= 0.15) return '#581c87'; // Dark purple
    if (value >= 0.10) return '#7c3aed'; // Purple
    if (value >= 0.07) return '#a78bfa'; // Light purple
    if (value >= 0.04) return '#c4b5fd'; // Very light purple
    if (value >= 0.02) return '#ddd6fe'; // Almost white purple
    return '#f3f4f6'; // Nearly white
  };

  const cellSize = 80; // Size of each cell in pixels

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900">
          Predictor correlation analysis
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Correlation matrix showing relationships between predictor variables
        </p>

        {/* Metric Selector */}
        {availableMetrics.length > 1 && (
          <div className="flex items-center gap-3">
            <label htmlFor="metric-select" className="text-sm font-medium text-gray-700">
              Correlation metric:
            </label>
            <select
              id="metric-select"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
            >
              {availableMetrics.map((metric) => {
                const label = metric === 'mutual_info'
                  ? 'Mutual information'
                  : metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' ');
                return (
                  <option key={metric} value={metric}>
                    {label}
                  </option>
                );
              })}
            </select>
          </div>
        )}
      </div>

      {/* Correlation Matrix */}
      <div className="overflow-x-auto overflow-y-hidden">
        <div className="inline-block">
          <div style={{ display: 'grid', gridTemplateColumns: `${cellSize}px repeat(${predictors.length}, ${cellSize}px)`, border: '1px solid #e5e7eb' }}>
            {/* Top-left empty cell */}
            <div style={{ width: cellSize, height: cellSize, borderRight: '1px solid #e5e7eb', borderBottom: '1px solid #e5e7eb' }} className="bg-white" />

            {/* Column headers */}
            {predictors.map((pred, idx) => (
              <div
                key={`header-${pred}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  borderRight: idx < predictors.length - 1 ? '1px solid #e5e7eb' : 'none',
                  borderBottom: '1px solid #e5e7eb'
                }}
                className="bg-gray-100 flex items-center justify-center font-semibold text-gray-900 text-sm"
              >
                <div
                  style={{
                    transform: 'rotate(-45deg)',
                    transformOrigin: 'center',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {pred}
                </div>
              </div>
            ))}

            {/* Rows */}
            {predictors.map((pred1, rowIdx) => (
              <>
                {/* Row header */}
                <div
                  key={`row-header-${pred1}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    borderRight: '1px solid #e5e7eb',
                    borderBottom: rowIdx < predictors.length - 1 ? '1px solid #e5e7eb' : 'none'
                  }}
                  className="bg-gray-100 flex items-center justify-center font-semibold text-gray-900 text-sm"
                >
                  {pred1}
                </div>

                {/* Correlation cells */}
                {predictors.map((pred2, colIdx) => {
                  const value = matrixData.get(pred1)?.get(pred2) ?? 0;
                  // Use purple scale for mutual_info, blue/red scale for correlations
                  const bgColor = selectedMetric === 'mutual_info' ? getMIColor(value) : getColor(value);
                  const textColor = selectedMetric === 'mutual_info'
                    ? (value > 0.07 ? '#ffffff' : '#000000')
                    : (Math.abs(value) > 0.5 ? '#ffffff' : '#000000');

                  return (
                    <div
                      key={`cell-${pred1}-${pred2}`}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        backgroundColor: bgColor,
                        color: textColor,
                        borderRight: colIdx < predictors.length - 1 ? '1px solid #e5e7eb' : 'none',
                        borderBottom: rowIdx < predictors.length - 1 ? '1px solid #e5e7eb' : 'none',
                      }}
                      className="flex items-center justify-center text-xs font-medium"
                      title={`${pred1} vs ${pred2}: ${value.toFixed(3)}`}
                    >
                      {selectedMetric === 'mutual_info' ? value.toFixed(3) : value.toFixed(2)}
                    </div>
                  );
                })}
              </>
            ))}
          </div>
        </div>
      </div>

      {/* Legend - only for correlation metrics (not mutual_info) */}
      {selectedMetric !== 'mutual_info' && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm text-gray-700 mb-3">
            <strong>Interpretation:</strong> Correlation values range from -1 to 1. Positive values (blue) indicate variables that increase together, negative values (red) indicate variables that move in opposite directions, and values near 0 (white) indicate little to no linear relationship.
          </p>
          <div className="flex items-center gap-4 mb-3">
            <span className="text-xs text-gray-600 font-medium">Color scale:</span>
            <div className="flex items-center gap-1">
              <div className="w-8 h-4 bg-red-500" title="-1.0 to -0.7" />
              <div className="w-8 h-4 bg-red-400" title="-0.7 to -0.4" />
              <div className="w-8 h-4 bg-red-300" title="-0.4 to -0.2" />
              <div className="w-8 h-4 bg-gray-100" title="-0.2 to 0.2" />
              <div className="w-8 h-4 bg-blue-300" title="0.2 to 0.4" />
              <div className="w-8 h-4 bg-blue-400" title="0.4 to 0.7" />
              <div className="w-8 h-4 bg-blue-600" title="0.7 to 1.0" />
            </div>
            <span className="text-xs text-gray-600">
              <span className="text-red-500">◄ Negative</span>
              <span className="mx-2">|</span>
              <span className="text-blue-500">Positive ►</span>
            </span>
          </div>
          <div className="pt-3 border-t border-blue-300">
            <p className="text-sm text-gray-700">
              <strong>Pearson vs Spearman:</strong> Pearson correlation measures linear relationships between variables and is sensitive to outliers. Spearman correlation measures monotonic relationships (whether variables consistently increase or decrease together) by ranking the data first, making it more robust to outliers and non-linear but monotonic relationships. Use Pearson for linear relationships and Spearman when the relationship may be non-linear or when data contains outliers.
            </p>
          </div>
        </div>
      )}

      {/* Predictor-Target Mutual Information Section */}
      <div className="mt-8 pt-8 border-t-2 border-gray-200">
        <h3 className="text-xl font-semibold mb-4 text-gray-900">
          Predictor-imputed variable mutual information
        </h3>

        {hasPredictorTargetMI ? (
          <>
            <p className="text-sm text-gray-600 mb-4">
              Mutual information between predictor variables and imputed target variables
            </p>

            {/* MI Matrix */}
            <div className="overflow-x-auto overflow-y-hidden mb-4">
              <div className="inline-block">
                <div style={{ display: 'grid', gridTemplateColumns: `${cellSize}px repeat(${targetsList.length}, ${cellSize}px)`, border: '1px solid #e5e7eb' }}>
                  {/* Top-left empty cell */}
                  <div style={{ width: cellSize, height: cellSize, borderRight: '1px solid #e5e7eb', borderBottom: '1px solid #e5e7eb' }} className="bg-white" />

                  {/* Column headers (targets) */}
                  {targetsList.map((target, idx) => (
                    <div
                      key={`header-${target}`}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        borderRight: idx < targetsList.length - 1 ? '1px solid #e5e7eb' : 'none',
                        borderBottom: '1px solid #e5e7eb'
                      }}
                      className="bg-gray-100 flex items-center justify-center font-semibold text-gray-900 text-sm"
                    >
                      <div
                        style={{
                          transform: 'rotate(-45deg)',
                          transformOrigin: 'center',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {target}
                      </div>
                    </div>
                  ))}

                  {/* Rows */}
                  {predictorsList.map((predictor, rowIdx) => (
                    <>
                      {/* Row header */}
                      <div
                        key={`row-header-${predictor}`}
                        style={{
                          width: cellSize,
                          height: cellSize,
                          borderRight: '1px solid #e5e7eb',
                          borderBottom: rowIdx < predictorsList.length - 1 ? '1px solid #e5e7eb' : 'none'
                        }}
                        className="bg-gray-100 flex items-center justify-center font-semibold text-gray-900 text-sm"
                      >
                        {predictor}
                      </div>

                      {/* MI cells */}
                      {targetsList.map((target, colIdx) => {
                        const value = miMatrixData.get(predictor)?.get(target) ?? 0;
                        const bgColor = getMIColor(value);
                        const textColor = value > 0.07 ? '#ffffff' : '#000000';

                        return (
                          <div
                            key={`cell-${predictor}-${target}`}
                            style={{
                              width: cellSize,
                              height: cellSize,
                              backgroundColor: bgColor,
                              color: textColor,
                              borderRight: colIdx < targetsList.length - 1 ? '1px solid #e5e7eb' : 'none',
                              borderBottom: rowIdx < predictorsList.length - 1 ? '1px solid #e5e7eb' : 'none',
                            }}
                            className="flex items-center justify-center text-xs font-medium"
                            title={`${predictor} → ${target}: ${value.toFixed(4)}`}
                          >
                            {value.toFixed(3)}
                          </div>
                        );
                      })}
                    </>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : null}

        {/* Explanation box - always shown */}
        <div className="p-4 bg-purple-50 border border-purple-200 rounded-md">
          <p className="text-sm text-gray-700 mb-2">
            <strong>What is mutual information?</strong> Mutual information measures how much information one variable provides about another. Unlike correlation, it captures both linear and non-linear relationships between variables. Values range from 0 (independent variables) to higher positive values (strong dependency).
          </p>
          <p className="text-sm text-gray-700 mb-3">
            <strong>Why measure it for imputed variables?</strong> Mutual information between predictors and imputed variables reveals which predictors are most informative for imputation. High mutual information indicates that a predictor strongly influences the imputed variable&apos;s distribution, making it crucial for accurate imputation. This helps validate that your imputation models are using the most relevant predictors and can identify when key predictive relationships exist in your data.
          </p>

          {/* Color scale within explanation box */}
          <div className="mt-3 pt-3 border-t border-purple-300">
            <div className="flex items-center gap-4">
              <span className="text-xs text-gray-600 font-medium">Color scale:</span>
              <div className="flex items-center gap-1">
                <div className="w-8 h-4 bg-gray-100" title="0 to 0.02" />
                <div className="w-8 h-4" style={{ backgroundColor: '#ddd6fe' }} title="0.02 to 0.04" />
                <div className="w-8 h-4" style={{ backgroundColor: '#c4b5fd' }} title="0.04 to 0.07" />
                <div className="w-8 h-4" style={{ backgroundColor: '#a78bfa' }} title="0.07 to 0.10" />
                <div className="w-8 h-4" style={{ backgroundColor: '#7c3aed' }} title="0.10 to 0.15" />
                <div className="w-8 h-4" style={{ backgroundColor: '#581c87' }} title="0.15+" />
              </div>
              <span className="text-xs text-gray-600">
                <span className="text-gray-400">Weak</span>
                <span className="mx-2">→</span>
                <span className="text-purple-700">Strong ►</span>
              </span>
            </div>
          </div>
        </div>

        {/* Message when no predictor-target data is available */}
        {!hasPredictorTargetMI && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-300 rounded-md">
            <p className="text-sm text-gray-700">
              <strong>Note:</strong> No predictor-imputed variable mutual information data was found in this CSV file. It is recommended to include this data in your analysis to understand which predictors are most informative for imputing each variable. This helps validate that your imputation models are leveraging the most relevant predictive relationships in your data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
