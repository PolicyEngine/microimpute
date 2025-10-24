'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import FileUpload from '@/components/FileUpload';
import VisualizationDashboard from '@/components/VisualizationDashboard';
import { parseImputationCSV } from '@/utils/csvParser';
import { ImputationDataPoint } from '@/types/imputation';
import { parseDeeplinkParams, GitHubArtifactInfo } from '@/utils/deeplinks';

function HomeContent() {
  const [data, setData] = useState<ImputationDataPoint[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [showDashboard, setShowDashboard] = useState(false);
  const [isLoadingFromDeeplink, setIsLoadingFromDeeplink] = useState(false);
  const [githubArtifactInfo, setGithubArtifactInfo] = useState<GitHubArtifactInfo | null>(null);

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
    } catch (error) {
      console.error('Error parsing CSV:', error);
      alert('Failed to parse CSV file. Please check the file format.');
    }
  };

  const handleViewDashboard = () => {
    if (data.length > 0) {
      setShowDashboard(true);
    }
  };

  const handleBackToUpload = () => {
    setShowDashboard(false);
    setData([]);
    setFileName('');
    setGithubArtifactInfo(null);
  };

  const handleDeeplinkLoadComplete = (primary: GitHubArtifactInfo | null) => {
    setIsLoadingFromDeeplink(false);
    if (primary) {
      setGithubArtifactInfo(primary);
      setShowDashboard(true);
    }
  };

  const handleGithubLoad = (primary: GitHubArtifactInfo | null) => {
    if (primary) {
      setGithubArtifactInfo(primary);
    }
  };


  return (
    <main className="min-h-screen bg-gray-50">
      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!showDashboard ? (
          <FileUpload
            onFileLoad={handleFileLoad}
            onViewDashboard={handleViewDashboard}
            deeplinkParams={deeplinkParams}
            isLoadingFromDeeplink={isLoadingFromDeeplink}
            onDeeplinkLoadComplete={handleDeeplinkLoadComplete}
            onGithubLoad={handleGithubLoad}
          />
        ) : (
          <VisualizationDashboard
            data={data}
            fileName={fileName}
            githubArtifactInfo={githubArtifactInfo}
            onBackToUpload={handleBackToUpload}
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

export default function Home() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    }>
      <HomeContent />
    </Suspense>
  );
}
