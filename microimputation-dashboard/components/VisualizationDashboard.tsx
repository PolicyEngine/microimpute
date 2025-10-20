'use client';

import { ImputationDataPoint } from '@/types/imputation';
import { GitHubArtifactInfo } from '@/utils/deeplinks';

interface VisualizationDashboardProps {
  data: ImputationDataPoint[];
  fileName: string;
  comparisonData?: {
    data: ImputationDataPoint[];
    filename: string;
  };
  githubArtifactInfo?: {
    primary: GitHubArtifactInfo | null;
    secondary?: GitHubArtifactInfo | null;
  } | null;
}

export default function VisualizationDashboard({
  data,
  fileName,
  comparisonData,
  githubArtifactInfo
}: VisualizationDashboardProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-12">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Visualization Dashboard</h2>
        <p className="text-xl text-gray-600 mb-2">Coming Soon...</p>
        <p className="text-gray-500">
          The visualization components for microimputation results will be implemented here.
        </p>
        <div className="mt-8 p-4 bg-gray-50 rounded">
          <p className="text-sm text-gray-600">
            Successfully loaded: <strong>{fileName}</strong>
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Records: <strong>{data.length}</strong>
          </p>
          {comparisonData && (
            <p className="text-sm text-gray-600 mt-1">
              Comparison file: <strong>{comparisonData.filename}</strong> ({comparisonData.data.length} records)
            </p>
          )}
        </div>
      </div>
    </div>
  );
}