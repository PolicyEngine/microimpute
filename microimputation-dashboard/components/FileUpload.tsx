'use client';

import { useState, useEffect, useCallback } from 'react';
import { Upload, File as FileIcon, Link, Database, GitBranch } from 'lucide-react';
import JSZip from 'jszip';
import { DeeplinkParams, GitHubArtifactInfo } from '@/utils/deeplinks';

interface FileUploadProps {
  onFileLoad: (content: string, filename: string) => void;
  onViewDashboard: () => void;
  onCompareLoad?: (content1: string, filename1: string, content2: string, filename2: string) => void;
  deeplinkParams?: DeeplinkParams | null;
  isLoadingFromDeeplink?: boolean;
  onDeeplinkLoadComplete?: (primary: GitHubArtifactInfo | null, secondary?: GitHubArtifactInfo | null | undefined) => void;
  onGithubLoad?: (primary: GitHubArtifactInfo | null, secondary?: GitHubArtifactInfo | null) => void;
}

interface GitHubCommit {
  sha: string;
  commit: {
    message: string;
    author: {
      date: string;
    };
  };
}

interface GitHubBranch {
  name: string;
  commit: {
    sha: string;
  };
}

interface GitHubArtifact {
  id: number;
  name: string;
  archive_download_url: string;
  size_in_bytes: number;
  created_at: string;
}

export default function FileUpload({
  onFileLoad,
  onViewDashboard,
  onCompareLoad,
  deeplinkParams,
  isLoadingFromDeeplink,
  onDeeplinkLoadComplete,
  onGithubLoad
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [activeTab, setActiveTab] = useState<'drop' | 'url' | 'sample' | 'github'>('drop');
  const [loadedFile, setLoadedFile] = useState<string>('');
  const [error, setError] = useState<string>('');

  // GitHub-specific state
  const [githubRepo, setGithubRepo] = useState('');
  const [githubBranches, setGithubBranches] = useState<GitHubBranch[]>([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [githubCommits, setGithubCommits] = useState<GitHubCommit[]>([]);
  const [selectedCommit, setSelectedCommit] = useState('');
  const [availableArtifacts, setAvailableArtifacts] = useState<GitHubArtifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState('');
  const [isLoadingGithubData, setIsLoadingGithubData] = useState(false);

  // Comparison mode state
  const [comparisonMode, setComparisonMode] = useState(false);
  const [selectedSecondBranch, setSelectedSecondBranch] = useState('');
  const [secondCommits, setSecondCommits] = useState<GitHubCommit[]>([]);
  const [selectedSecondCommit, setSelectedSecondCommit] = useState('');
  const [secondArtifacts, setSecondArtifacts] = useState<GitHubArtifact[]>([]);
  const [selectedSecondArtifact, setSelectedSecondArtifact] = useState('');

  // Helper function to load a single artifact from deeplink parameters
  const loadArtifactFromDeeplink = useCallback(async (artifactInfo: GitHubArtifactInfo, githubToken: string): Promise<string> => {
    // First, get the artifacts for the specific commit
    const [owner, repo] = artifactInfo.repo.split('/');
    const runsResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}/actions/runs?head_sha=${artifactInfo.commit}`, {
      headers: {
        'Authorization': `Bearer ${githubToken}`,
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'PolicyEngine-Dashboard/1.0'
      }
    });

    if (!runsResponse.ok) {
      throw new Error(`Failed to fetch workflow runs: ${runsResponse.status} ${runsResponse.statusText}`);
    }

    const runsData = await runsResponse.json();
    const completedRuns = runsData.workflow_runs.filter((run: { status: string }) => run.status === 'completed');

    if (completedRuns.length === 0) {
      throw new Error('No completed workflow runs found for this commit');
    }

    // Find the artifact by name
    let targetArtifact = null;
    for (const run of completedRuns) {
      const artifactsResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}/actions/runs/${run.id}/artifacts`, {
        headers: {
          'Authorization': `Bearer ${githubToken}`,
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'PolicyEngine-Dashboard/1.0'
        }
      });

      if (artifactsResponse.ok) {
        const artifactsData = await artifactsResponse.json();
        targetArtifact = artifactsData.artifacts.find((artifact: { name: string }) => artifact.name === artifactInfo.artifact);
        if (targetArtifact) break;
      }
    }

    if (!targetArtifact) {
      throw new Error(`Artifact "${artifactInfo.artifact}" not found for commit ${artifactInfo.commit}`);
    }

    // Download and extract the artifact
    const downloadResponse = await fetch(targetArtifact.archive_download_url, {
      headers: {
        'Authorization': `Bearer ${githubToken}`,
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'PolicyEngine-Dashboard/1.0'
      }
    });

    if (!downloadResponse.ok) {
      throw new Error(`Failed to download artifact: ${downloadResponse.status} ${downloadResponse.statusText}`);
    }

    const zipBuffer = await downloadResponse.arrayBuffer();
    const zip = new JSZip();
    const zipContents = await zip.loadAsync(zipBuffer);

    // Find CSV files in the zip
    const csvFiles = Object.keys(zipContents.files).filter(filename => filename.endsWith('.csv'));

    if (csvFiles.length === 0) {
      throw new Error('No CSV files found in the artifact');
    }

    // Use the first CSV file found
    const csvFile = zipContents.files[csvFiles[0]];
    const csvContent = await csvFile.async('text');

    // Validate CSV format
    validateCSVFormat(csvContent);

    return csvContent;
  }, []);

  // Load GitHub artifacts directly from deeplink parameters
  const loadDeeplinkArtifacts = useCallback(async (primary: GitHubArtifactInfo, secondary?: GitHubArtifactInfo) => {
    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setError('🔄 Loading data from GitHub artifacts...');

      // Load primary artifact
      const primaryData = await loadArtifactFromDeeplink(primary, githubToken);

      if (secondary && onCompareLoad) {
        // Load secondary artifact for comparison
        const secondaryData = await loadArtifactFromDeeplink(secondary, githubToken);

        // Generate display names with commit info
        const primaryDisplayName = `${primary.repo}@${primary.branch} (${primary.commit.substring(0, 7)}) - ${primary.artifact}`;
        const secondaryDisplayName = `${secondary.repo}@${secondary.branch} (${secondary.commit.substring(0, 7)}) - ${secondary.artifact}`;

        onCompareLoad(primaryData, primaryDisplayName, secondaryData, secondaryDisplayName);
        setLoadedFile(`Comparison: ${primaryDisplayName} vs ${secondaryDisplayName}`);
      } else {
        // Single artifact load
        const displayName = `${primary.repo}@${primary.branch} (${primary.commit.substring(0, 7)}) - ${primary.artifact}`;
        onFileLoad(primaryData, displayName);
        setLoadedFile(displayName);
      }

      // Notify parent component that deeplink loading is complete
      if (onDeeplinkLoadComplete) {
        onDeeplinkLoadComplete(primary, secondary);
      }

      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load artifacts from deeplink');
      if (onDeeplinkLoadComplete) {
        onDeeplinkLoadComplete(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, [onFileLoad, onCompareLoad, onDeeplinkLoadComplete, loadArtifactFromDeeplink]);

  // Handle deeplink loading on mount
  useEffect(() => {
    if (deeplinkParams && isLoadingFromDeeplink) {
      setActiveTab('github');

      if (deeplinkParams.mode === 'comparison' && deeplinkParams.primary && deeplinkParams.secondary) {
        setComparisonMode(true);
        setGithubRepo(deeplinkParams.primary.repo);
        setSelectedBranch(deeplinkParams.primary.branch);
        setSelectedCommit(deeplinkParams.primary.commit);
        setSelectedArtifact(deeplinkParams.primary.artifact);
        setSelectedSecondBranch(deeplinkParams.secondary.branch);
        setSelectedSecondCommit(deeplinkParams.secondary.commit);
        setSelectedSecondArtifact(deeplinkParams.secondary.artifact);

        // Auto-load comparison data
        loadDeeplinkArtifacts(deeplinkParams.primary, deeplinkParams.secondary);
      } else if (deeplinkParams.primary) {
        setComparisonMode(false);
        setGithubRepo(deeplinkParams.primary.repo);
        setSelectedBranch(deeplinkParams.primary.branch);
        setSelectedCommit(deeplinkParams.primary.commit);
        setSelectedArtifact(deeplinkParams.primary.artifact);

        // Auto-load single artifact data
        loadDeeplinkArtifacts(deeplinkParams.primary);
      }
    }
  }, [deeplinkParams, isLoadingFromDeeplink, loadDeeplinkArtifacts]);

  function validateCSVFormat(content: string): void {
    // Expected column names for microimputation dashboard
    const EXPECTED_COLUMNS = [
      'type',
      'method',
      'variable',
      'quantile',
      'metric_name',
      'metric_value',
      'split',
      'additional_info'
    ];

    const lines = content.trim().split('\n');

    // Parse header
    const header = lines[0].split(',').map(col => col.trim());

    // Check if all expected columns are present
    const missingColumns = EXPECTED_COLUMNS.filter(col => !header.includes(col));
    if (missingColumns.length > 0) {
      throw new Error(
        `Invalid CSV format: Missing required columns: ${missingColumns.join(', ')}. ` +
        `Expected columns: ${EXPECTED_COLUMNS.join(', ')}`
      );
    }

    // Check for extra columns
    const extraColumns = header.filter(col => !EXPECTED_COLUMNS.includes(col));
    if (extraColumns.length > 0) {
      throw new Error(
        `Invalid CSV format: Unexpected columns found: ${extraColumns.join(', ')}. ` +
        `Expected columns: ${EXPECTED_COLUMNS.join(', ')}`
      );
    }

    // Check column order matches expected
    const columnOrderMismatch = EXPECTED_COLUMNS.some((col, idx) => header[idx] !== col);
    if (columnOrderMismatch) {
      throw new Error(
        `Invalid CSV format: Columns are not in the expected order. ` +
        `Expected order: ${EXPECTED_COLUMNS.join(', ')}`
      );
    }

    // Find the index of additional_info column
    const additionalInfoIndex = header.indexOf('additional_info');

    // Validate JSON format in additional_info column for a sample of rows
    // We'll check the first 10 data rows to avoid processing large files
    const maxRowsToCheck = Math.min(10, lines.length - 1);

    for (let i = 1; i <= maxRowsToCheck; i++) {
      const line = lines[i];
      if (!line.trim()) continue; // Skip empty lines

      // Parse CSV row (basic parsing - doesn't handle quoted commas)
      const columns = line.split(',');

      if (columns.length !== EXPECTED_COLUMNS.length) {
        throw new Error(
          `Invalid CSV format: Row ${i + 1} has ${columns.length} columns, ` +
          `but ${EXPECTED_COLUMNS.length} columns are expected.`
        );
      }

      let additionalInfo = columns[additionalInfoIndex]?.trim();

      if (additionalInfo) {
        // Handle CSV quote escaping: pandas doubles quotes when writing to CSV
        // e.g., {"key": "value"} becomes "{""key"": ""value""}" in the CSV
        // Remove outer quotes if present and unescape inner quotes
        if (additionalInfo.startsWith('"') && additionalInfo.endsWith('"')) {
          additionalInfo = additionalInfo.slice(1, -1); // Remove outer quotes
          additionalInfo = additionalInfo.replace(/""/g, '"'); // Unescape doubled quotes
        }

        try {
          JSON.parse(additionalInfo);
        } catch (e) {
          throw new Error(
            `Invalid CSV format: Row ${i + 1} contains invalid JSON in the 'additional_info' column. ` +
            `Value: "${additionalInfo.substring(0, 50)}${additionalInfo.length > 50 ? '...' : ''}"`
          );
        }
      }
    }
  }

  async function processFile(file: globalThis.File) {
    setIsLoading(true);
    try {
      const content = await file.text();

      // Basic CSV validation
      if (!content.trim()) {
        throw new Error('The file appears to be empty or contains only whitespace.');
      }

      // Check for basic CSV structure
      const lines = content.trim().split('\n');
      if (lines.length < 2) {
        throw new Error('The file must contain at least a header row and one data row.');
      }

      // Validate CSV format (columns and JSON)
      validateCSVFormat(content);

      onFileLoad(content, file.name);
      setLoadedFile(file.name);
    } catch (err) {
      setError(
        err instanceof Error
          ? `File processing error: ${err.message}`
          : 'Failed to read file. Please ensure it is a valid CSV file and try again.'
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
    setError('');

    const files = Array.from(e.dataTransfer.files);

    if (files.length === 0) {
      setError('No files were dropped. Please try again.');
      return;
    }

    if (files.length > 1) {
      setError('Please drop only one file at a time. Multiple files are not supported.');
      return;
    }

    const file = files[0];

    if (!file.name.endsWith('.csv')) {
      setError(`Invalid file type: "${file.name}". Please drop a CSV file (.csv extension required).`);
      return;
    }

    if (file.size === 0) {
      setError('The dropped file appears to be empty. Please check your file and try again.');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      // 500 MB limit
      setError('File is too large (over 500 MB). Please use a smaller CSV file.');
      return;
    }

    processFile(file);
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    setError('');

    if (!file) {
      setError('No file was selected. Please try again.');
      return;
    }

    if (!file.name.endsWith('.csv')) {
      setError(`Invalid file type: "${file.name}". Please select a CSV file (.csv extension required).`);
      return;
    }

    if (file.size === 0) {
      setError('The selected file appears to be empty. Please check your file and try again.');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      setError('File is too large (over 500 MB). Please select a smaller CSV file.');
      return;
    }

    processFile(file);
  }

  async function handleUrlLoad() {
    if (!urlInput.trim()) {
      setError('Please enter a URL to load a CSV file.');
      return;
    }

    let url: URL;
    try {
      url = new URL(urlInput.trim());
    } catch {
      setError('Invalid URL format. Please enter a valid URL (e.g., https://example.com/data.csv).');
      return;
    }

    if (!url.pathname.toLowerCase().endsWith('.csv') && !urlInput.toLowerCase().includes('csv')) {
      setError('URL should point to a CSV file. Please ensure the URL ends with .csv or contains CSV data.');
      return;
    }

    if (url.protocol !== 'https:' && url.protocol !== 'http:') {
      setError('Only HTTP and HTTPS URLs are supported. Please use a web URL.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30_000); // 30 s timeout

      const response = await fetch(urlInput.trim(), {
        signal: controller.signal,
        headers: { Accept: 'text/csv, text/plain, */*' }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const status = response.status;
        const map: Record<number, string> = {
          404: 'File not found (404). Please check the URL and try again.',
          403: 'Access forbidden (403). The server denied access to this file.',
          401: 'Authentication required (401). This file requires login credentials.',
          500: 'Server error (500). The remote server encountered an error.',
          503: 'Service unavailable (503). The server is temporarily unavailable.'
        };
        throw new Error(map[status] || `HTTP error ${status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type') || '';
      if (
        !contentType.includes('text/csv') &&
        !contentType.includes('text/plain') &&
        !contentType.includes('application/csv')
      ) {
        setError(`Warning: Server returned content type "${contentType}". This may not be a CSV file.`);
      }

      const content = await response.text();

      if (!content.trim()) {
        throw new Error('The downloaded file appears to be empty.');
      }

      const lines = content.trim().split('\n');
      if (lines.length < 2) {
        throw new Error('The downloaded file must contain at least a header row and one data row.');
      }

      // Validate CSV format (columns and JSON)
      validateCSVFormat(content);

      const filename = url.pathname.split('/').pop() || 'remote-file.csv';
      onFileLoad(content, filename);
      setLoadedFile(filename);
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Request timed out after 30 s. Please check your connection and try again.');
        } else if (err.message.includes('Failed to fetch')) {
          setError('Network error: Unable to reach the URL. Please check the address and your connection.');
        } else {
          setError(`Failed to load from URL: ${err.message}`);
        }
      } else {
        setError('Unknown error occurred while loading the URL. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSampleLoad() {
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/microimputation_results.csv');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const content = await response.text();

      // Validate CSV format (columns and JSON)
      validateCSVFormat(content);

      onFileLoad(content, 'microimputation_results.csv');
      setLoadedFile('microimputation_results.csv');
    } catch (err) {
      setError(`Failed to load sample data: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  }

  async function fetchGithubBranches() {
    if (!githubRepo.trim()) {
      setError('Please enter a GitHub repository (e.g., owner/repo)');
      return;
    }

    const repoMatch = githubRepo.trim().match(/^([^/]+)\/([^/]+)$/);
    if (!repoMatch) {
      setError('Invalid repository format. Use "owner/repo" format (e.g., "PolicyEngine/microimpute")');
      return;
    }

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoadingGithubData(true);
    setError('');

    try {
      // Fetch all branches with pagination support
      const allBranches: GitHubBranch[] = [];
      let page = 1;
      const perPage = 100; // Maximum allowed by GitHub API

      while (true) {
        const response = await fetch(`https://api.github.com/repos/${githubRepo}/branches?per_page=${perPage}&page=${page}`, {
          headers: {
            'Authorization': `Bearer ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PolicyEngine-Dashboard/1.0'
          }
        });

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Repository not found. Please check the repository name and ensure it is accessible.');
          } else if (response.status === 403) {
            throw new Error('Access forbidden. Please check your GitHub token permissions or repository access.');
          }
          throw new Error(`Failed to fetch branches: ${response.status} ${response.statusText}`);
        }

        const branches: GitHubBranch[] = await response.json();

        if (branches.length === 0) {
          // No more branches to fetch
          break;
        }

        allBranches.push(...branches);

        // If we got fewer branches than requested, we've reached the end
        if (branches.length < perPage) {
          break;
        }

        page++;

        // Safety check to prevent infinite loops (GitHub repos rarely have more than 1000 branches)
        if (page > 10) {
          console.warn('Stopped fetching branches after 10 pages (1000 branches) to prevent excessive API calls');
          break;
        }
      }

      setGithubBranches(allBranches);

      // Auto-select main/master branch if available
      const defaultBranch = allBranches.find(b => b.name === 'main' || b.name === 'master');
      if (defaultBranch) {
        setSelectedBranch(defaultBranch.name);
        await fetchGithubCommits(defaultBranch.name);
      }
    } catch (err) {
      setError(`GitHub API error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingGithubData(false);
    }
  }

  async function fetchGithubCommits(branch: string) {
    if (!githubRepo.trim() || !branch) return;

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoadingGithubData(true);
    try {
      const response = await fetch(`https://api.github.com/repos/${githubRepo}/commits?sha=${branch}&per_page=20`, {
        headers: {
          'Authorization': `Bearer ${githubToken}`,
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'PolicyEngine-Dashboard/1.0'
        }
      });
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Branch not found or repository is private.');
        } else if (response.status === 403) {
          throw new Error('Access forbidden. Please check your GitHub token permissions or repository access.');
        }
        throw new Error(`Failed to fetch commits: ${response.status} ${response.statusText}`);
      }

      const commits: GitHubCommit[] = await response.json();
      setGithubCommits(commits);

      // Auto-select latest commit and fetch its artifacts
      if (commits.length > 0) {
        setSelectedCommit(commits[0].sha);
        await fetchGithubArtifacts(commits[0].sha);
      }
    } catch (err) {
      setError(`GitHub API error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingGithubData(false);
    }
  }

  async function fetchGithubArtifacts(commitSha: string) {
    if (!githubRepo.trim() || !commitSha) return;

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoadingGithubData(true);
    setAvailableArtifacts([]);
    setSelectedArtifact('');

    try {
      const [owner, repo] = githubRepo.split('/');

      // Get workflow runs for the commit
      const runsResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/actions/runs?head_sha=${commitSha}`,
        {
          headers: {
            'Authorization': `Bearer ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PolicyEngine-Dashboard/1.0'
          }
        }
      );

      if (!runsResponse.ok) {
        if (runsResponse.status === 403) {
          throw new Error(`GitHub API rate limit exceeded or token permissions insufficient (403). Please try again later or check your token permissions.`);
        } else if (runsResponse.status === 404) {
          throw new Error(`Repository or commit not found (404). Please check the repository name and commit SHA.`);
        } else {
          throw new Error(`Failed to fetch workflow runs: ${runsResponse.status} ${runsResponse.statusText}`);
        }
      }

      const runsData = await runsResponse.json();
      const runs = runsData.workflow_runs;

      if (!runs || runs.length === 0) {
        setError('No workflow runs found for this commit.');
        return;
      }

      // Collect all imputation artifacts from completed runs
      const allArtifacts: GitHubArtifact[] = [];

      for (const run of runs) {
        if (run.status !== 'completed') continue;

        try {
          const artifactsResponse = await fetch(
            `https://api.github.com/repos/${owner}/${repo}/actions/runs/${run.id}/artifacts`,
            {
              headers: {
                'Authorization': `Bearer ${githubToken}`,
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'PolicyEngine-Dashboard/1.0'
              }
            }
          );

          if (!artifactsResponse.ok) continue;

          const artifactsData = await artifactsResponse.json();
          const artifacts = artifactsData.artifacts;

          // Filter for imputation artifacts
          const imputationArtifacts = artifacts.filter((artifact: GitHubArtifact) =>
            artifact.name.toLowerCase().includes('impute') ||
            artifact.name.toLowerCase().includes('imputation') ||
            artifact.name.toLowerCase().includes('result') ||
            artifact.name.toLowerCase().includes('.csv')
          );

          allArtifacts.push(...imputationArtifacts);
        } catch {
          continue;
        }
      }

      if (allArtifacts.length === 0) {
        setError('No imputation artifacts found for this commit.');
        return;
      }

      // Remove duplicates and sort by creation date (newest first)
      const uniqueArtifacts = allArtifacts
        .filter((artifact, index, self) =>
          index === self.findIndex(a => a.name === artifact.name)
        )
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      setAvailableArtifacts(uniqueArtifacts);

      // Auto-select the first artifact
      if (uniqueArtifacts.length > 0) {
        setSelectedArtifact(uniqueArtifacts[0].id.toString());
      }

    } catch (err) {
      setError(`Failed to fetch artifacts: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingGithubData(false);
    }
  }

  async function loadGithubArtifact() {
    if (!selectedArtifact) {
      setError('Please select an artifact to load');
      return;
    }

    const artifact = availableArtifacts.find(a => a.id.toString() === selectedArtifact);
    if (!artifact) {
      setError('Selected artifact not found');
      return;
    }

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setError('🔄 Downloading and extracting CSV from artifact...');

      const downloadResponse = await fetch(artifact.archive_download_url, {
        headers: {
          'Authorization': `Bearer ${githubToken}`,
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'PolicyEngine-Dashboard/1.0'
        }
      });

      if (!downloadResponse.ok) {
        throw new Error(`Failed to download artifact: ${downloadResponse.status}`);
      }

      const zipBuffer = await downloadResponse.arrayBuffer();
      const zip = new JSZip();
      const zipContent = await zip.loadAsync(zipBuffer);

      // Find CSV files in the ZIP
      const csvFiles = Object.keys(zipContent.files).filter(filename =>
        filename.toLowerCase().endsWith('.csv') && !zipContent.files[filename].dir
      );

      if (csvFiles.length === 0) {
        throw new Error('No CSV files found in the artifact ZIP');
      }

      // Use the first CSV file found
      const csvFilename = csvFiles[0];
      const csvFile = zipContent.files[csvFilename];
      const csvContent = await csvFile.async('text');

      // Validate the CSV content
      if (!csvContent.trim()) {
        throw new Error('The extracted CSV file is empty');
      }

      // Check for basic CSV structure
      const lines = csvContent.trim().split('\n');
      if (lines.length < 2) {
        throw new Error('The CSV must contain at least a header row and one data row');
      }

      // Validate CSV format (columns and JSON)
      validateCSVFormat(csvContent);

      // Success! Load the CSV into the dashboard with commit info
      const commitShort = selectedCommit.slice(0, 8);
      const branchInfo = selectedBranch ? ` (${selectedBranch})` : '';
      const displayName = `${csvFilename} @ ${commitShort}${branchInfo}`;

      onFileLoad(csvContent, displayName);
      setLoadedFile(displayName);
      setError('');

      // Notify parent component about GitHub artifact info for sharing
      if (onGithubLoad) {
        const artifactInfo: GitHubArtifactInfo = {
          repo: githubRepo,
          branch: selectedBranch,
          commit: selectedCommit,
          artifact: artifact.name
        };
        onGithubLoad(artifactInfo, null);
      }

      // Clear the GitHub state since we successfully loaded the file
      setGithubRepo('');
      setGithubBranches([]);
      setSelectedBranch('');
      setGithubCommits([]);
      setSelectedCommit('');
      setAvailableArtifacts([]);
      setSelectedArtifact('');

    } catch (extractError) {
      console.error('CSV extraction error:', extractError);
      setError(`❌ Failed to extract CSV: ${extractError instanceof Error ? extractError.message : 'Unknown error'}. Try using the URL tab with a direct CSV link.`);
    } finally {
      setIsLoading(false);
    }
  }

  async function fetchSecondBranchCommits(branch: string) {
    if (!githubRepo.trim() || !branch) return;

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoadingGithubData(true);
    try {
      const response = await fetch(`https://api.github.com/repos/${githubRepo}/commits?sha=${branch}&per_page=20`, {
        headers: {
          'Authorization': `Bearer ${githubToken}`,
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'PolicyEngine-Dashboard/1.0'
        }
      });
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Branch not found or repository is private.');
        } else if (response.status === 403) {
          throw new Error('Access forbidden. Please check your GitHub token permissions or repository access.');
        }
        throw new Error(`Failed to fetch commits: ${response.status} ${response.statusText}`);
      }

      const commits: GitHubCommit[] = await response.json();
      setSecondCommits(commits);

      // Auto-select latest commit and fetch its artifacts
      if (commits.length > 0) {
        setSelectedSecondCommit(commits[0].sha);
        await fetchSecondArtifacts(commits[0].sha);
      }
    } catch (err) {
      setError(`GitHub API error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingGithubData(false);
    }
  }

  async function fetchSecondArtifacts(commitSha: string) {
    if (!githubRepo.trim() || !commitSha) return;

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoadingGithubData(true);
    setSecondArtifacts([]);
    setSelectedSecondArtifact('');

    try {
      const [owner, repo] = githubRepo.split('/');

      // Get workflow runs for the commit
      const runsResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/actions/runs?head_sha=${commitSha}`,
        {
          headers: {
            'Authorization': `Bearer ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PolicyEngine-Dashboard/1.0'
          }
        }
      );

      if (!runsResponse.ok) {
        if (runsResponse.status === 403) {
          throw new Error(`GitHub API rate limit exceeded or token permissions insufficient (403). Please try again later or check your token permissions.`);
        } else if (runsResponse.status === 404) {
          throw new Error(`Repository or commit not found (404). Please check the repository name and commit SHA.`);
        } else {
          throw new Error(`Failed to fetch workflow runs: ${runsResponse.status} ${runsResponse.statusText}`);
        }
      }

      const runsData = await runsResponse.json();
      const runs = runsData.workflow_runs;

      if (!runs || runs.length === 0) {
        setError('No workflow runs found for this commit.');
        return;
      }

      // Collect all imputation artifacts from completed runs
      const allArtifacts: GitHubArtifact[] = [];

      for (const run of runs) {
        if (run.status !== 'completed') continue;

        try {
          const artifactsResponse = await fetch(
            `https://api.github.com/repos/${owner}/${repo}/actions/runs/${run.id}/artifacts`,
            {
              headers: {
                'Authorization': `Bearer ${githubToken}`,
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'PolicyEngine-Dashboard/1.0'
              }
            }
          );

          if (!artifactsResponse.ok) continue;

          const artifactsData = await artifactsResponse.json();
          const artifacts = artifactsData.artifacts;

          // Filter for imputation artifacts
          const imputationArtifacts = artifacts.filter((artifact: GitHubArtifact) =>
            artifact.name.toLowerCase().includes('impute') ||
            artifact.name.toLowerCase().includes('imputation') ||
            artifact.name.toLowerCase().includes('result') ||
            artifact.name.toLowerCase().includes('.csv')
          );

          allArtifacts.push(...imputationArtifacts);
        } catch {
          continue;
        }
      }

      if (allArtifacts.length === 0) {
        setError('No imputation artifacts found for this commit.');
        return;
      }

      // Remove duplicates and sort by creation date (newest first)
      const uniqueArtifacts = allArtifacts
        .filter((artifact, index, self) =>
          index === self.findIndex(a => a.name === artifact.name)
        )
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      setSecondArtifacts(uniqueArtifacts);

      // Auto-select the first artifact
      if (uniqueArtifacts.length > 0) {
        setSelectedSecondArtifact(uniqueArtifacts[0].id.toString());
      }

    } catch (err) {
      setError(`Failed to fetch artifacts: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingGithubData(false);
    }
  }

  async function loadComparisonData() {
    if (!selectedArtifact || !selectedSecondArtifact || !onCompareLoad) {
      setError('Please select artifacts from both commits to compare');
      return;
    }

    const firstArtifact = availableArtifacts.find(a => a.id.toString() === selectedArtifact);
    const secondArtifact = secondArtifacts.find(a => a.id.toString() === selectedSecondArtifact);

    if (!firstArtifact || !secondArtifact) {
      setError('Selected artifacts not found');
      return;
    }

    const githubToken = process.env.NEXT_PUBLIC_GITHUB_TOKEN;
    if (!githubToken) {
      setError('GitHub token not configured. Please set NEXT_PUBLIC_GITHUB_TOKEN environment variable.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setError('🔄 Downloading and extracting CSV files for comparison...');

      // Download both artifacts
      const [firstDownload, secondDownload] = await Promise.all([
        fetch(firstArtifact.archive_download_url, {
          headers: {
            'Authorization': `Bearer ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PolicyEngine-Dashboard/1.0'
          }
        }),
        fetch(secondArtifact.archive_download_url, {
          headers: {
            'Authorization': `Bearer ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PolicyEngine-Dashboard/1.0'
          }
        })
      ]);

      if (!firstDownload.ok || !secondDownload.ok) {
        throw new Error('Failed to download one or both artifacts');
      }

      // Extract CSVs from both artifacts
      const [firstZipBuffer, secondZipBuffer] = await Promise.all([
        firstDownload.arrayBuffer(),
        secondDownload.arrayBuffer()
      ]);

      const firstZip = new JSZip();
      const secondZip = new JSZip();
      const [firstZipContent, secondZipContent] = await Promise.all([
        firstZip.loadAsync(firstZipBuffer),
        secondZip.loadAsync(secondZipBuffer)
      ]);

      // Find CSV files in both ZIPs
      const firstCsvFiles = Object.keys(firstZipContent.files).filter(filename =>
        filename.toLowerCase().endsWith('.csv') && !firstZipContent.files[filename].dir
      );
      const secondCsvFiles = Object.keys(secondZipContent.files).filter(filename =>
        filename.toLowerCase().endsWith('.csv') && !secondZipContent.files[filename].dir
      );

      if (firstCsvFiles.length === 0 || secondCsvFiles.length === 0) {
        throw new Error('No CSV files found in one or both artifacts');
      }

      // Extract CSV content
      const [firstCsvContent, secondCsvContent] = await Promise.all([
        firstZipContent.files[firstCsvFiles[0]].async('text'),
        secondZipContent.files[secondCsvFiles[0]].async('text')
      ]);

      // Create display names with commit info
      const firstCommitShort = selectedCommit.slice(0, 8);
      const secondCommitShort = selectedSecondCommit.slice(0, 8);

      const firstBranchInfo = selectedBranch !== selectedSecondBranch ? ` (${selectedBranch})` : '';
      const secondBranchInfo = selectedBranch !== selectedSecondBranch ? ` (${selectedSecondBranch})` : '';

      const firstName = `${firstCsvFiles[0]} @ ${firstCommitShort}${firstBranchInfo}`;
      const secondName = `${secondCsvFiles[0]} @ ${secondCommitShort}${secondBranchInfo}`;

      // Load into comparison mode
      onCompareLoad(firstCsvContent, firstName, secondCsvContent, secondName);

      // Notify parent component about GitHub artifact info for sharing
      if (onGithubLoad) {
        const primaryArtifactInfo: GitHubArtifactInfo = {
          repo: githubRepo,
          branch: selectedBranch,
          commit: selectedCommit,
          artifact: firstArtifact.name
        };
        const secondaryArtifactInfo: GitHubArtifactInfo = {
          repo: githubRepo,
          branch: selectedSecondBranch,
          commit: selectedSecondCommit,
          artifact: secondArtifact.name
        };
        onGithubLoad(primaryArtifactInfo, secondaryArtifactInfo);
      }

      setError('');

    } catch (extractError) {
      console.error('Comparison extraction error:', extractError);
      setError(`❌ Failed to extract comparison data: ${extractError instanceof Error ? extractError.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Load imputation data</h2>
        <p className="text-gray-600">Choose how you would like to load your CSV file</p>
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-md p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {loadedFile && (
        <div className="mb-4 bg-green-50 border border-green-200 rounded-md p-3">
          <p className="text-sm text-green-700">Successfully loaded: {loadedFile}</p>
        </div>
      )}

      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 mb-6">
        <button
          onClick={() => setActiveTab('drop')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'drop'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Upload className="w-4 h-4 inline mr-2" />
          Drop file
        </button>
        <button
          onClick={() => setActiveTab('url')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'url'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Link className="w-4 h-4 inline mr-2" />
          URL
        </button>
        <button
          onClick={() => setActiveTab('github')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'github'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <GitBranch className="w-4 h-4 inline mr-2" />
          GitHub
        </button>
        <button
          onClick={() => setActiveTab('sample')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'sample'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Database className="w-4 h-4 inline mr-2" />
          Sample data
        </button>
      </div>

      {/* Tab content */}
      {activeTab === 'drop' && (
        <div className="space-y-4">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <FileIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-900 mb-2">Drop your CSV file here</p>
            <p className="text-gray-600 mb-4">or click to browse files</p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileInput}
              className="hidden"
              id="file-input"
            />
            <label
              htmlFor="file-input"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer"
            >
              Choose file
            </label>
          </div>
        </div>
      )}

      {activeTab === 'url' && (
        <div className="space-y-4">
          <div>
            <label htmlFor="url-input" className="block text-sm font-medium text-gray-700 mb-2">
              CSV file URL
            </label>
            <div className="flex space-x-2">
              <input
                id="url-input"
                type="url"
                value={urlInput}
                onChange={e => setUrlInput(e.target.value)}
                placeholder="https://example.com/data.csv"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleUrlLoad}
                disabled={isLoading || !urlInput.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Loading...' : 'Load'}
              </button>
            </div>
            <p className="text-sm text-gray-500 mt-2">Enter a direct URL to a CSV file accessible via HTTP/HTTPS</p>
          </div>
        </div>
      )}

      {activeTab === 'sample' && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-start space-x-3">
              <Database className="w-6 h-6 text-blue-600 mt-1" />
              <div className="flex-1">
                <h3 className="text-lg font-medium text-blue-900 mb-2">Load sample imputation data</h3>
                <p className="text-blue-700 mb-4">
                  Try the dashboard with sample microimputation results to explore the visualization features.
                </p>
                <ul className="text-sm text-blue-600 mb-4 space-y-1">
                  <li>• Shows imputation results for various variables</li>
                  <li>• Benchmarks different imputation methods</li>
                  <li>• Includes validation metrics</li>
                </ul>
                <button
                  onClick={handleSampleLoad}
                  disabled={isLoading}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Loading Sample...' : 'Load sample data'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'github' && (
        <div className="space-y-6">
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
            <div className="flex items-start space-x-3">
              <GitBranch className="w-6 h-6 text-gray-600 mt-1" />
              <div className="flex-1">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Load from GitHub repository
                </h3>
                <p className="text-gray-700 mb-4">
                  Load imputation data from GitHub Actions artifacts in public repositories.
                </p>

                {/* Repository Input */}
                <div className="mb-4">
                  <label htmlFor="github-repo" className="block text-sm font-medium text-gray-700 mb-2">
                    Repository (owner/repo)
                  </label>
                  <div className="flex space-x-2">
                    <input
                      id="github-repo"
                      type="text"
                      value={githubRepo}
                      onChange={(e) => setGithubRepo(e.target.value)}
                      placeholder="PolicyEngine/microimpute"
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      onClick={fetchGithubBranches}
                      disabled={isLoadingGithubData || !githubRepo.trim()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                    >
                      {isLoadingGithubData ? 'Loading...' : 'Fetch'}
                    </button>
                  </div>
                </div>

                {/* Comparison Mode Toggle */}
                <div className="mb-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={comparisonMode}
                      onChange={(e) => {
                        setComparisonMode(e.target.checked);
                        if (!e.target.checked) {
                          // Reset second selection states
                          setSelectedSecondBranch('');
                          setSecondCommits([]);
                          setSelectedSecondCommit('');
                          setSecondArtifacts([]);
                          setSelectedSecondArtifact('');
                        }
                      }}
                      className="mr-2 rounded"
                    />
                    <span className="text-sm font-medium text-gray-700">
                      Compare two imputation runs (different branches/commits)
                    </span>
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    Enable this to compare imputation results between different branches or commits
                  </p>
                </div>

                {/* Branch Selection */}
                {githubBranches.length > 0 && (
                  <div className="mb-4">
                    <label htmlFor="github-branch" className="block text-sm font-medium text-gray-700 mb-2">
                      Branch
                    </label>
                    <select
                      id="github-branch"
                      value={selectedBranch}
                      onChange={(e) => {
                        setSelectedBranch(e.target.value);
                        fetchGithubCommits(e.target.value);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select a branch</option>
                      {githubBranches.map((branch) => (
                        <option key={branch.name} value={branch.name}>
                          {branch.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Commit Selection */}
                {githubCommits.length > 0 && (
                  <div className="mb-4">
                    <label htmlFor="github-commit" className="block text-sm font-medium text-gray-700 mb-2">
                      Commit
                    </label>
                    <select
                      id="github-commit"
                      value={selectedCommit}
                      onChange={(e) => {
                        setSelectedCommit(e.target.value);
                        fetchGithubArtifacts(e.target.value);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select a commit</option>
                      {githubCommits.map((commit) => (
                        <option key={commit.sha} value={commit.sha}>
                          {commit.sha.slice(0, 8)} - {commit.commit.message.slice(0, 60)}
                          {commit.commit.message.length > 60 ? '...' : ''}
                        </option>
                      ))}
                    </select>
                    {selectedCommit && (
                      <p className="text-sm text-gray-500 mt-1">
                        {githubCommits.find(c => c.sha === selectedCommit)?.commit.author.date &&
                          new Date(githubCommits.find(c => c.sha === selectedCommit)!.commit.author.date).toLocaleString()
                        }
                      </p>
                    )}
                  </div>
                )}

                {/* Artifact Selection */}
                {availableArtifacts.length > 0 && (
                  <div className="mb-4">
                    <label htmlFor="github-artifact" className="block text-sm font-medium text-gray-700 mb-2">
                      Artifact ({availableArtifacts.length} available)
                    </label>
                    <select
                      id="github-artifact"
                      value={selectedArtifact}
                      onChange={(e) => setSelectedArtifact(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select an artifact</option>
                      {availableArtifacts.map((artifact) => (
                        <option key={artifact.id} value={artifact.id.toString()}>
                          {artifact.name} ({(artifact.size_in_bytes / 1024).toFixed(1)} KB)
                        </option>
                      ))}
                    </select>
                    {selectedArtifact && (
                      <p className="text-sm text-gray-500 mt-1">
                        {availableArtifacts.find(a => a.id.toString() === selectedArtifact)?.created_at &&
                          `Created: ${new Date(availableArtifacts.find(a => a.id.toString() === selectedArtifact)!.created_at).toLocaleString()}`
                        }
                      </p>
                    )}
                  </div>
                )}

                {/* Second Selection for Comparison */}
                {comparisonMode && githubBranches.length > 0 && (
                  <div className="border-t border-gray-200 pt-4 mt-6">
                    <h4 className="text-md font-semibold text-gray-800 mb-4">Second Imputation Run (for comparison)</h4>

                    {/* Second Branch Selection */}
                    <div className="mb-4">
                      <label htmlFor="github-second-branch" className="block text-sm font-medium text-gray-700 mb-2">
                        Branch
                      </label>
                      <select
                        id="github-second-branch"
                        value={selectedSecondBranch}
                        onChange={(e) => {
                          setSelectedSecondBranch(e.target.value);
                          fetchSecondBranchCommits(e.target.value);
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">Select a branch</option>
                        {githubBranches.map((branch) => (
                          <option key={`second-${branch.name}`} value={branch.name}>
                            {branch.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Second Commit Selection */}
                    {secondCommits.length > 0 && (
                      <div className="mb-4">
                        <label htmlFor="github-second-commit" className="block text-sm font-medium text-gray-700 mb-2">
                          Commit
                        </label>
                        <select
                          id="github-second-commit"
                          value={selectedSecondCommit}
                          onChange={(e) => {
                            setSelectedSecondCommit(e.target.value);
                            fetchSecondArtifacts(e.target.value);
                          }}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">Select a commit</option>
                          {secondCommits.map((commit) => (
                            <option key={`second-${commit.sha}`} value={commit.sha}>
                              {commit.sha.slice(0, 8)} - {commit.commit.message.slice(0, 60)}
                              {commit.commit.message.length > 60 ? '...' : ''}
                            </option>
                          ))}
                        </select>
                        {selectedSecondCommit && (
                          <p className="text-sm text-gray-500 mt-1">
                            {secondCommits.find(c => c.sha === selectedSecondCommit)?.commit.author.date &&
                              new Date(secondCommits.find(c => c.sha === selectedSecondCommit)!.commit.author.date).toLocaleString()
                            }
                          </p>
                        )}
                      </div>
                    )}

                    {/* Second Artifact Selection */}
                    {secondArtifacts.length > 0 && (
                      <div className="mb-4">
                        <label htmlFor="github-second-artifact" className="block text-sm font-medium text-gray-700 mb-2">
                          Artifact ({secondArtifacts.length} available)
                        </label>
                        <select
                          id="github-second-artifact"
                          value={selectedSecondArtifact}
                          onChange={(e) => setSelectedSecondArtifact(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">Select an artifact</option>
                          {secondArtifacts.map((artifact) => (
                            <option key={`second-${artifact.id}`} value={artifact.id.toString()}>
                              {artifact.name} ({(artifact.size_in_bytes / 1024).toFixed(1)} KB)
                            </option>
                          ))}
                        </select>
                        {selectedSecondArtifact && (
                          <p className="text-sm text-gray-500 mt-1">
                            {secondArtifacts.find(a => a.id.toString() === selectedSecondArtifact)?.created_at &&
                              `Created: ${new Date(secondArtifacts.find(a => a.id.toString() === selectedSecondArtifact)!.created_at).toLocaleString()}`
                            }
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Load Button */}
                {selectedArtifact && (!comparisonMode || selectedSecondArtifact) && (
                  <button
                    onClick={comparisonMode ? loadComparisonData : loadGithubArtifact}
                    disabled={isLoading}
                    className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md disabled:opacity-50"
                  >
                    {isLoading
                      ? (comparisonMode ? 'Loading comparison...' : 'Loading artifact...')
                      : (comparisonMode ? 'Compare imputation runs' : 'Load imputation data')
                    }
                  </button>
                )}

                <div className="mt-4 text-sm text-gray-600">
                  <p className="mb-2">📌 <strong>Note:</strong> This feature finds imputation CSV files in GitHub Actions artifacts.</p>
                  <ul className="space-y-1 text-xs">
                    <li>• ✅ Works with PolicyEngine public repositories</li>
                    <li>• ✅ Authenticated access to download artifacts</li>
                    <li>• ✅ Automatically finds imputation results from CI/CD runs</li>
                    <li>• ✅ Full CSV extraction from ZIP artifacts</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* View Dashboard Button */}
      {loadedFile && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <button
            onClick={onViewDashboard}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-4 rounded-md transition-colors"
          >
            View dashboard
          </button>
        </div>
      )}

      {/* Global loading indicator */}
      {(isLoading || isLoadingGithubData) && (
        <div className="mt-4 text-center">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
          <p className="text-sm text-gray-600 mt-2">
            {isLoadingGithubData ? 'Loading GitHub data...' : 'Loading file...'}
          </p>
        </div>
      )}
    </div>
  );
}