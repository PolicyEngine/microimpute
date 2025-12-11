'use client';

import { useMemo, useState } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import { GitHubArtifactInfo, createShareableUrl } from '@/utils/deeplinks';
import BenchmarkLossCharts from './BenchmarkLossCharts';
import PerVariableCharts from './PerVariableCharts';
import VisualizationTabs from './VisualizationTabs';
import PredictorCorrelationMatrix from './PredictorCorrelationMatrix';
import PredictorOrderingRobustness from './PredictorOrderingRobustness';
import ImputationResults from './ImputationResults';
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

    // Get all unique methods from benchmark data
    const allMethods = hasBenchmarkLoss
      ? Array.from(new Set(data.filter(d => d.type === 'benchmark_loss').map(d => d.method)))
      : [];

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

    // Check for actual distribution distance data (wasserstein or kl_divergence)
    const distributionData = data.filter(d => d.type === 'distribution_distance');
    const hasWasserstein = distributionData.some(d => d.metric_name === 'wasserstein_distance' && d.metric_value !== null);
    const hasKLDivergence = distributionData.some(d => d.metric_name === 'kl_divergence' && d.metric_value !== null);
    const hasDistributionDistance = hasWasserstein || hasKLDivergence;

    // Check for predictor correlation data
    const correlationData = data.filter(d => d.type === 'predictor_correlation');
    const hasPredictorCorrelation = correlationData.length > 0 && correlationData.some(d => d.metric_value !== null);

    // Check for predictor ordering/importance data
    const progressiveData = data.filter(d => d.type === 'progressive_inclusion');
    const importanceData = data.filter(d => d.type === 'predictor_importance');
    const hasPredictorOrdering = (progressiveData.length > 0 && progressiveData.some(d => d.metric_value !== null)) ||
                                   (importanceData.length > 0 && importanceData.some(d => d.metric_value !== null));

    // Find imputed variables (from distribution_distance data)
    const imputedVars = new Set<string>();
    distributionData.forEach(d => {
      if (d.variable && d.metric_value !== null) {
        imputedVars.add(d.variable);
      }
    });

    // Calculate best performing model (same logic as BenchmarkLossCharts)
    let bestModel = '';

    if (hasBenchmarkLoss) {
      const benchmarkData = data.filter(d => d.type === 'benchmark_loss');
      const methods = Array.from(new Set(benchmarkData.map(d => d.method)));

      // Filter quantile and log loss data (matching BenchmarkLossCharts logic)
      const quantileLossData = benchmarkData.filter(
        d => d.metric_name === 'quantile_loss' &&
             d.split === 'test' &&
             typeof d.quantile === 'number' &&
             d.quantile >= 0 &&
             d.quantile <= 1
      );

      const logLossData = benchmarkData.filter(
        d => d.metric_name === 'log_loss' &&
             d.split === 'test' &&
             d.metric_value !== null
      );

      // Calculate average quantile loss per method
      const quantileLossAvg = new Map<string, number>();
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

      // Calculate average log loss per method
      const logLossAvg = new Map<string, number>();
      const logLossVarCounts = new Map<string, Set<string>>();

      if (logLossData.length > 0) {
        const methodSums = new Map<string, { sum: number; count: number }>();
        logLossData.forEach(d => {
          if (d.metric_value !== null) {
            if (!methodSums.has(d.method)) {
              methodSums.set(d.method, { sum: 0, count: 0 });
            }
            const entry = methodSums.get(d.method)!;
            entry.sum += d.metric_value;
            entry.count += 1;

            if (!logLossVarCounts.has(d.method)) {
              logLossVarCounts.set(d.method, new Set());
            }
            logLossVarCounts.get(d.method)!.add(d.variable);
          }
        });
        methodSums.forEach((value, method) => {
          logLossAvg.set(method, value.sum / value.count);
        });
      }

      // Rank methods by each metric (lower is better)
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

      // Calculate weighted combined rank
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
      let bestRank = Infinity;
      combinedRanks.forEach((rank, method) => {
        if (rank < bestRank) {
          bestRank = rank;
          bestModel = method;
        }
      });
    }

    // Calculate quality scores by variable for model performance
    let modelExcellent = 0;
    let modelGood = 0;
    let modelPoor = 0;
    let modelScore = 0;
    let modelQuality = '';

    if (hasBenchmarkLoss && bestModel) {
      const benchmarkData = data.filter(d => d.type === 'benchmark_loss');
      const bestModelVars = benchmarkData.filter(
        d => d.method === bestModel && d.split === 'test' &&
             d.quantile === 'mean' && !d.variable.includes('_mean_all') && d.metric_value !== null
      );

      bestModelVars.forEach(d => {
        const loss = d.metric_value ?? 0;
        if (loss < 0.02) modelExcellent++;
        else if (loss < 0.05) modelGood++;
        else modelPoor++;
      });

      const totalModelVars = modelExcellent + modelGood + modelPoor;
      if (totalModelVars > 0) {
        modelScore = ((modelExcellent * 100) + (modelGood * 75)) / totalModelVars;
        if (modelScore >= 90) modelQuality = 'Excellent';
        else if (modelScore >= 70) modelQuality = 'Good';
        else modelQuality = 'Needs improvement';
      }
    }

    // Calculate quality scores by variable for distributional accuracy
    let distExcellent = 0;
    let distGood = 0;
    let distPoor = 0;
    let distScore = 0;
    let distQuality = '';

    distributionData.forEach(d => {
      const value = d.metric_value ?? 0;
      // Different thresholds for Wasserstein vs KL-divergence
      if (d.metric_name === 'wasserstein_distance') {
        if (value < 0.01) distExcellent++;
        else if (value < 0.05) distGood++;
        else distPoor++;
      } else if (d.metric_name === 'kl_divergence') {
        if (value < 0.1) distExcellent++;
        else if (value < 0.5) distGood++;
        else distPoor++;
      }
    });

    const totalDistVars = distExcellent + distGood + distPoor;
    if (totalDistVars > 0) {
      distScore = ((distExcellent * 100) + (distGood * 75)) / totalDistVars;
      if (distScore >= 90) distQuality = 'Excellent';
      else if (distScore >= 70) distQuality = 'Good';
      else distQuality = 'Needs improvement';
    }

    // Calculate overall quality (weighted average)
    let overallScore = 0;
    let overallQuality = '';
    let overallColor = '';
    const hasModelScore = modelScore > 0;
    const hasDistScore = distScore > 0;

    if (hasModelScore && hasDistScore) {
      overallScore = (modelScore + distScore) / 2;
    } else if (hasModelScore) {
      overallScore = modelScore;
    } else if (hasDistScore) {
      overallScore = distScore;
    }

    if (overallScore >= 90) {
      overallQuality = 'Excellent quality';
      overallColor = 'text-green-700 bg-green-50 border-green-500';
    } else if (overallScore >= 70) {
      overallQuality = 'Good quality';
      overallColor = 'text-yellow-700 bg-yellow-50 border-yellow-500';
    } else if (overallScore > 0) {
      overallQuality = 'Needs improvement';
      overallColor = 'text-red-700 bg-red-50 border-red-500';
    }

    return {
      hasBenchmarkLoss,
      hasDistributionDistance,
      hasPredictorCorrelation,
      hasPredictorOrdering,
      numericalVars,
      categoricalVars,
      hasPerVariableData: numericalVars.length > 0 || categoricalVars.length > 0,
      imputedVars: Array.from(imputedVars).sort(),
      bestModel,
      overallScore,
      overallQuality,
      overallColor,
      modelScore,
      modelQuality,
      modelExcellent,
      modelGood,
      modelPoor,
      distScore,
      distQuality,
      distExcellent,
      distGood,
      distPoor,
      allMethods,
    };
  }, [data]);

  // Build tabs based on available data
  const tabs = useMemo(() => {
    const tabsList = [];

    if (dataAnalysis.hasBenchmarkLoss) {
      tabsList.push({ id: 'overview', label: 'Model benchmarking' });
    }

    if (dataAnalysis.hasDistributionDistance) {
      tabsList.push({
        id: 'imputation',
        label: 'Imputation results',
      });
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

    if (dataAnalysis.hasPredictorOrdering) {
      tabsList.push({
        id: 'ordering',
        label: 'Predictor selection',
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

      {/* Imputation Summary */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-1">Imputation summary</h2>
        <p className="text-sm text-gray-600 mb-4">
          Assessment of the quality of the imputations produced by the best-performing (or the only selected) model
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mt-6">
          {/* Imputed Variables Section - 1/4 width */}
          <div className="border border-gray-200 rounded-md p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide">
              Imputed Variables
            </h3>
            {dataAnalysis.imputedVars.length > 0 ? (
              <div className="space-y-2">
                <p className="text-xs text-gray-600 mb-2">
                  {dataAnalysis.imputedVars.length} variable{dataAnalysis.imputedVars.length !== 1 ? 's' : ''} imputed
                </p>
                <ul className={`space-y-1 ${dataAnalysis.imputedVars.length > 3 ? 'max-h-32 overflow-y-auto' : ''}`}>
                  {dataAnalysis.imputedVars.map((variable) => (
                    <li key={variable} className="text-sm font-mono text-gray-900 bg-gray-50 px-2 py-1 rounded">
                      {variable}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-sm text-gray-500 italic">
                No imputed variable information available in the CSV
              </p>
            )}
          </div>

          {/* Best Model Section - 1/4 width */}
          <div className="border border-gray-200 rounded-md p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide">
              {dataAnalysis.allMethods.length === 1 ? 'Imputation Model' : 'Best Performing Model'}
            </h3>
            {dataAnalysis.bestModel ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-base font-bold text-blue-700">
                    {dataAnalysis.bestModel}
                  </span>
                  {dataAnalysis.allMethods.length === 1 && (
                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                      Only model
                    </span>
                  )}
                  {dataAnalysis.allMethods.length > 1 && (
                    <span className="text-xs text-green-700 bg-green-50 px-2 py-0.5 rounded">
                      Best of {dataAnalysis.allMethods.length}
                    </span>
                  )}
                </div>
                {dataAnalysis.allMethods.length > 1 && (
                  <p className="text-xs text-gray-600">
                    Selected based on combined performance across all cross-validation loss metrics
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500 italic">
                No model information available in the CSV
              </p>
            )}
          </div>

          {/* Metrics Section - 1/2 width */}
          <div className="lg:col-span-2 border border-gray-200 rounded-md p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">
              Performance Metrics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {/* Average Test Losses */}
              {(() => {
                const benchmarkData = data.filter(d => d.type === 'benchmark_loss' && d.method === dataAnalysis.bestModel && d.split === 'test');

                // Calculate avg quantile loss
                const quantileLossData = benchmarkData.filter(
                  d => d.metric_name === 'quantile_loss' &&
                       typeof d.quantile === 'number' &&
                       d.metric_value !== null
                );
                const avgQuantileLoss = quantileLossData.length > 0
                  ? quantileLossData.reduce((sum, d) => sum + (d.metric_value ?? 0), 0) / quantileLossData.length
                  : null;

                // Calculate avg log loss
                const logLossData = benchmarkData.filter(
                  d => d.metric_name === 'log_loss' &&
                       d.metric_value !== null
                );
                const avgLogLoss = logLossData.length > 0
                  ? logLossData.reduce((sum, d) => sum + (d.metric_value ?? 0), 0) / logLossData.length
                  : null;

                // Calculate avg Wasserstein distance
                const wassersteinData = data.filter(
                  d => d.type === 'distribution_distance' &&
                       d.metric_name === 'wasserstein_distance' &&
                       d.metric_value !== null
                );
                const avgWasserstein = wassersteinData.length > 0
                  ? wassersteinData.reduce((sum, d) => sum + (d.metric_value ?? 0), 0) / wassersteinData.length
                  : null;

                // Calculate avg KL divergence
                const klData = data.filter(
                  d => d.type === 'distribution_distance' &&
                       d.metric_name === 'kl_divergence' &&
                       d.metric_value !== null
                );
                const avgKL = klData.length > 0
                  ? klData.reduce((sum, d) => sum + (d.metric_value ?? 0), 0) / klData.length
                  : null;

                return (
                  <>
                    {avgQuantileLoss !== null && (
                      <div className="bg-purple-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-1">Avg. test quantile loss</p>
                        <p className="text-lg font-bold text-gray-900">{avgQuantileLoss.toFixed(4)}</p>
                      </div>
                    )}
                    {avgLogLoss !== null && (
                      <div className="bg-purple-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-1">Avg. test log loss</p>
                        <p className="text-lg font-bold text-gray-900">{avgLogLoss.toFixed(4)}</p>
                      </div>
                    )}
                    {avgWasserstein !== null && (
                      <div className="bg-orange-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-1">Avg. wasserstein distance</p>
                        <p className="text-lg font-bold text-gray-900">{avgWasserstein.toFixed(4)}</p>
                      </div>
                    )}
                    {avgKL !== null && (
                      <div className="bg-orange-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-1">Avg. KL divergence</p>
                        <p className="text-lg font-bold text-gray-900">{avgKL.toFixed(4)}</p>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          </div>
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

        {/* Predictor Ordering and Robustness Tab */}
        {activeTab === 'ordering' && (
          <PredictorOrderingRobustness data={data} />
        )}

        {/* Imputation Results Tab */}
        {activeTab === 'imputation' && (
          <ImputationResults data={data} />
        )}
      </div>
    </div>
  );
}