'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import FileUpload from '@/components/FileUpload';
import VisualizationDashboard from '@/components/VisualizationDashboard';
import { parseImputationCSV } from '@/utils/csvParser';
import { ImputationDataPoint } from '@/types/imputation';
import { parseDeeplinkParams, GitHubArtifactInfo } from '@/utils/deeplinks';

export default function Home() {
  const [data, setData] = useState<ImputationDataPoint[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [showDashboard, setShowDashboard] = useState(false);
  const [isLoadingFromDeeplink, setIsLoadingFromDeeplink] = useState(false);
  const [githubArtifactInfo, setGithubArtifactInfo] = useState<{
    primary: GitHubArtifactInfo | null;
    secondary?: GitHubArtifactInfo | null;
  } | null>(null);

  // Comparison mode state
  const [comparisonData, setComparisonData] = useState<{
    data1: ImputationDataPoint[];
    filename1: string;
    data2: ImputationDataPoint[];
    filename2: string;
  } | null>(null);

  const searchParams = useSearchParams();
  const deeplinkParams = parseDeeplinkParams(searchParams);

  useEffect(() => {
    if (deeplinkParams) {
      setIsLoadingFromDeeplink(true);
    }
  }, [deeplinkParams]);

  const handleFileLoad = (csvContent: string, filename: string) => {
    try {
      const parsedData = parseImputationCSV(csvContent);
      setData(parsedData);
      setFileName(filename);
      setComparisonData(null); // Clear comparison data when loading single file
    } catch (error) {
      console.error('Error parsing CSV:', error);
      alert('Failed to parse CSV file. Please check the file format.');
    }
  };

  const handleCompareLoad = (content1: string, filename1: string, content2: string, filename2: string) => {
    try {
      const data1 = parseImputationCSV(content1);
      const data2 = parseImputationCSV(content2);
      setComparisonData({
        data1,
        filename1,
        data2,
        filename2
      });
      setData([]); // Clear single data when loading comparison
    } catch (error) {
      console.error('Error parsing comparison CSVs:', error);
      alert('Failed to parse one or both CSV files. Please check the file formats.');
    }
  };

  const handleViewDashboard = () => {
    if (data.length > 0 || comparisonData) {
      setShowDashboard(true);
    }
  };

  const handleBackToUpload = () => {
    setShowDashboard(false);
    setData([]);
    setFileName('');
    setComparisonData(null);
    setGithubArtifactInfo(null);
  };

  const handleDeeplinkLoadComplete = (primary: GitHubArtifactInfo | null, secondary?: GitHubArtifactInfo | null) => {
    setIsLoadingFromDeeplink(false);
    if (primary) {
      setGithubArtifactInfo({ primary, secondary: secondary || undefined });
      setShowDashboard(true);
    }
  };

  const handleGithubLoad = (primary: GitHubArtifactInfo | null, secondary?: GitHubArtifactInfo | null) => {
    if (primary) {
      setGithubArtifactInfo({ primary, secondary: secondary || undefined });
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <h1 className="text-3xl font-bold text-gray-900">
                MicroImpute Dashboard
              </h1>
              <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded">
                Beta
              </span>
            </div>
            {showDashboard && (
              <button
                onClick={handleBackToUpload}
                className="text-sm text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Back to upload
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!showDashboard ? (
          <FileUpload
            onFileLoad={handleFileLoad}
            onViewDashboard={handleViewDashboard}
            onCompareLoad={handleCompareLoad}
            deeplinkParams={deeplinkParams}
            isLoadingFromDeeplink={isLoadingFromDeeplink}
            onDeeplinkLoadComplete={handleDeeplinkLoadComplete}
            onGithubLoad={handleGithubLoad}
          />
        ) : (
          <VisualizationDashboard
            data={comparisonData ? comparisonData.data1 : data}
            fileName={comparisonData ? comparisonData.filename1 : fileName}
            comparisonData={comparisonData ? {
              data: comparisonData.data2,
              filename: comparisonData.filename2
            } : undefined}
            githubArtifactInfo={githubArtifactInfo}
          />
        )}
      </div>

      {/* Footer */}
      <footer className="mt-16 bg-white border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>© 2024 PolicyEngine. Part of the MicroImpute suite.</p>
            <p className="mt-2">
              <a
                href="https://github.com/PolicyEngine/microimpute"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800"
              >
                View on GitHub
              </a>
              {' • '}
              <a
                href="https://policyengine.org"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800"
              >
                PolicyEngine.org
              </a>
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
