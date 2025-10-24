'use client';

import { useMemo, useState } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import { GitHubArtifactInfo } from '@/utils/deeplinks';
import BenchmarkLossCharts from './BenchmarkLossCharts';
import PerVariableCharts from './PerVariableCharts';
import VisualizationTabs from './VisualizationTabs';

interface VisualizationDashboardProps {
  data: ImputationDataPoint[];
  fileName: string;
  githubArtifactInfo?: GitHubArtifactInfo | null;
}

export default function VisualizationDashboard({
  data,
  fileName,
}: VisualizationDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');

  // Analyze data structure and available visualizations
  const dataAnalysis = useMemo(() => {
    const types = new Set(data.map(d => d.type));
    const hasBenchmarkLoss = types.has('benchmark_loss');

    // Extract individual variables (not aggregates)
    const numericalVars: string[] = [];
    const categoricalVars: string[] = [];

    if (hasBenchmarkLoss) {
      const benchmarkData = data.filter(d => d.type === 'benchmark_loss');

      // Find variables with quantile_loss (numerical)
      const qlVars = new Set(
        benchmarkData
          .filter(d =>
            d.metric_name === 'quantile_loss' &&
            !d.variable.includes('_mean_all')
          )
          .map(d => d.variable)
      );
      numericalVars.push(...Array.from(qlVars));

      // Find variables with log_loss (categorical)
      const llVars = new Set(
        benchmarkData
          .filter(d =>
            d.metric_name === 'log_loss' &&
            !d.variable.includes('_mean_all') &&
            d.metric_value !== null
          )
          .map(d => d.variable)
      );
      categoricalVars.push(...Array.from(llVars));
    }

    return {
      hasBenchmarkLoss,
      hasDistributionDistance: types.has('distribution_distance'),
      hasPredictorCorrelation: types.has('predictor_correlation'),
      numericalVars,
      categoricalVars,
      hasPerVariableData: numericalVars.length > 0 || categoricalVars.length > 0,
    };
  }, [data]);

  // Build tabs based on available data
  const tabs = useMemo(() => {
    const tabsList = [];

    if (dataAnalysis.hasBenchmarkLoss) {
      tabsList.push({ id: 'overview', label: 'Overview' });
    }

    if (dataAnalysis.numericalVars.length > 0) {
      tabsList.push({
        id: 'numerical',
        label: 'Numerical Variables',
        count: dataAnalysis.numericalVars.length,
      });
    }

    if (dataAnalysis.categoricalVars.length > 0) {
      tabsList.push({
        id: 'categorical',
        label: 'Categorical Variables',
        count: dataAnalysis.categoricalVars.length,
      });
    }

    return tabsList;
  }, [dataAnalysis]);

  if (!dataAnalysis.hasBenchmarkLoss) {
    return (
      <div className="space-y-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Visualization Dashboard</h2>
          <div className="p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">
              Successfully loaded: <strong>{fileName}</strong> ({data.length} records)
            </p>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-lg p-12">
          <div className="text-center">
            <p className="text-xl text-gray-600 mb-2">No visualization data found</p>
            <p className="text-gray-500">
              Upload a CSV file with benchmark_loss data to see visualizations.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Visualization Dashboard</h2>
        <div className="p-4 bg-gray-50 rounded">
          <p className="text-sm text-gray-600">
            Successfully loaded: <strong>{fileName}</strong>
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Records: <strong>{data.length}</strong>
          </p>
          {dataAnalysis.numericalVars.length > 0 && (
            <p className="text-sm text-gray-600 mt-1">
              Numerical variables: <strong>{dataAnalysis.numericalVars.length}</strong>
            </p>
          )}
          {dataAnalysis.categoricalVars.length > 0 && (
            <p className="text-sm text-gray-600 mt-1">
              Categorical variables: <strong>{dataAnalysis.categoricalVars.length}</strong>
            </p>
          )}
        </div>
      </div>

      {/* Tabs Navigation */}
      {tabs.length > 1 && (
        <div className="bg-white rounded-lg shadow-lg px-6 pt-6">
          <VisualizationTabs
            tabs={tabs}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        </div>
      )}

      {/* Tab Content */}
      <div>
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <BenchmarkLossCharts data={data} />
        )}

        {/* Numerical Variables Tab */}
        {activeTab === 'numerical' && (
          <div className="space-y-8">
            {dataAnalysis.numericalVars.map((variable) => (
              <div key={variable} className="bg-white p-6 rounded-lg shadow">
                <PerVariableCharts
                  data={data}
                  variable={variable}
                  metricType="quantile_loss"
                />
              </div>
            ))}
          </div>
        )}

        {/* Categorical Variables Tab */}
        {activeTab === 'categorical' && (
          <div className="space-y-8">
            {dataAnalysis.categoricalVars.map((variable) => (
              <div key={variable} className="bg-white p-6 rounded-lg shadow">
                <PerVariableCharts
                  data={data}
                  variable={variable}
                  metricType="log_loss"
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}