'use client';

import { useMemo, useState } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import { GitHubArtifactInfo, createShareableUrl } from '@/utils/deeplinks';
import BenchmarkLossCharts from './BenchmarkLossCharts';
import PerVariableCharts from './PerVariableCharts';
import VisualizationTabs from './VisualizationTabs';
import PredictorCorrelationMatrix from './PredictorCorrelationMatrix';
import { Share } from 'lucide-react';

interface VisualizationDashboardProps {
  data: ImputationDataPoint[];
  fileName: string;
  githubArtifactInfo?: GitHubArtifactInfo | null;
  onBackToUpload: () => void;
}

export default function VisualizationDashboard({
  data,
  fileName,
  githubArtifactInfo,
  onBackToUpload,
}: VisualizationDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');

  // Handle sharing the dashboard via deeplink
  const handleShare = async () => {
    if (!githubArtifactInfo) return;

    try {
      const shareUrl = createShareableUrl(githubArtifactInfo);
      await navigator.clipboard.writeText(shareUrl);
      alert('Shareable URL copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy URL:', err);
      alert('Failed to copy URL to clipboard');
    }
  };

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
      tabsList.push({ id: 'overview', label: 'Model benchmarking' });
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

    if (dataAnalysis.hasPredictorCorrelation) {
      tabsList.push({
        id: 'correlation',
        label: 'Predictor correlation',
      });
    }

    return tabsList;
  }, [dataAnalysis]);

  if (!dataAnalysis.hasBenchmarkLoss) {
    return (
      <div className="space-y-8">
        {/* Header */}
        <div>
          <div className="flex justify-between items-start mb-4 gap-4">
            <div className="flex-1 min-w-0">
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Microimpute Dashboard</h1>
              <p className="text-gray-600 break-words">
                Loaded: <span className="text-blue-600 break-all">{fileName}</span>
              </p>
            </div>
            <div className="flex gap-3 flex-shrink-0">
              {githubArtifactInfo && (
                <button
                  onClick={handleShare}
                  className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition-colors flex items-center gap-2 whitespace-nowrap"
                >
                  <Share size={16} />
                  Share Dashboard
                </button>
              )}
              <button
                onClick={onBackToUpload}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors whitespace-nowrap"
              >
                Load new file
              </button>
            </div>
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
      <div>
        <div className="flex justify-between items-start mb-4 gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Microimpute Dashboard</h1>
            <p className="text-gray-600 break-words">
              Loaded: <span className="text-blue-600 break-all">{fileName}</span>
            </p>
          </div>
          <div className="flex gap-3 flex-shrink-0">
            {githubArtifactInfo && (
              <button
                onClick={handleShare}
                className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition-colors flex items-center gap-2 whitespace-nowrap"
              >
                <Share size={16} />
                Share Dashboard
              </button>
            )}
            <button
              onClick={onBackToUpload}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors whitespace-nowrap"
            >
              Load new file
            </button>
          </div>
        </div>
      </div>

      {/* Data Info */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-3">Dataset Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">Total Records</p>
            <p className="text-2xl font-bold text-gray-900">{data.length}</p>
          </div>
          {dataAnalysis.numericalVars.length > 0 && (
            <div className="p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Numerical Variables</p>
              <p className="text-2xl font-bold text-gray-900">{dataAnalysis.numericalVars.length}</p>
            </div>
          )}
          {dataAnalysis.categoricalVars.length > 0 && (
            <div className="p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Categorical Variables</p>
              <p className="text-2xl font-bold text-gray-900">{dataAnalysis.categoricalVars.length}</p>
            </div>
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

        {/* Predictor Correlation Tab */}
        {activeTab === 'correlation' && (
          <PredictorCorrelationMatrix data={data} />
        )}
      </div>
    </div>
  );
}