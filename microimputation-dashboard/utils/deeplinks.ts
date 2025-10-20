// Deeplink utilities for GitHub artifact sharing

export interface GitHubArtifactInfo {
  repo: string;
  branch: string;
  commit: string;
  artifact: string;
}

export interface DeeplinkParams {
  mode?: 'single' | 'comparison';
  primary?: GitHubArtifactInfo;
  secondary?: GitHubArtifactInfo;
}

export function parseDeeplinkParams(searchParams: URLSearchParams): DeeplinkParams | null {
  const mode = searchParams.get('mode') || 'single';

  const primaryRepo = searchParams.get('repo');
  const primaryBranch = searchParams.get('branch');
  const primaryCommit = searchParams.get('commit');
  const primaryArtifact = searchParams.get('artifact');

  if (!primaryRepo || !primaryBranch || !primaryCommit || !primaryArtifact) {
    // Check for comparison mode parameters
    const repo1 = searchParams.get('repo1');
    const branch1 = searchParams.get('branch1');
    const commit1 = searchParams.get('commit1');
    const artifact1 = searchParams.get('artifact1');

    const repo2 = searchParams.get('repo2');
    const branch2 = searchParams.get('branch2');
    const commit2 = searchParams.get('commit2');
    const artifact2 = searchParams.get('artifact2');

    if (repo1 && branch1 && commit1 && artifact1 && repo2 && branch2 && commit2 && artifact2) {
      return {
        mode: 'comparison',
        primary: {
          repo: repo1,
          branch: branch1,
          commit: commit1,
          artifact: artifact1,
        },
        secondary: {
          repo: repo2,
          branch: branch2,
          commit: commit2,
          artifact: artifact2,
        },
      };
    }

    return null;
  }

  const params: DeeplinkParams = {
    mode: mode as 'single' | 'comparison',
    primary: {
      repo: primaryRepo,
      branch: primaryBranch,
      commit: primaryCommit,
      artifact: primaryArtifact,
    },
  };

  // Check for secondary parameters for comparison mode
  const secondaryRepo = searchParams.get('repo2') || primaryRepo;
  const secondaryBranch = searchParams.get('branch2');
  const secondaryCommit = searchParams.get('commit2');
  const secondaryArtifact = searchParams.get('artifact2');

  if (secondaryBranch && secondaryCommit && secondaryArtifact) {
    params.mode = 'comparison';
    params.secondary = {
      repo: secondaryRepo,
      branch: secondaryBranch,
      commit: secondaryCommit,
      artifact: secondaryArtifact,
    };
  }

  return params;
}

export function createShareableUrl(baseUrl: string, artifactInfo: GitHubArtifactInfo, secondaryInfo?: GitHubArtifactInfo): string {
  const url = new URL(baseUrl);

  if (secondaryInfo) {
    // Comparison mode
    url.searchParams.set('mode', 'comparison');
    url.searchParams.set('repo1', artifactInfo.repo);
    url.searchParams.set('branch1', artifactInfo.branch);
    url.searchParams.set('commit1', artifactInfo.commit);
    url.searchParams.set('artifact1', artifactInfo.artifact);
    url.searchParams.set('repo2', secondaryInfo.repo);
    url.searchParams.set('branch2', secondaryInfo.branch);
    url.searchParams.set('commit2', secondaryInfo.commit);
    url.searchParams.set('artifact2', secondaryInfo.artifact);
  } else {
    // Single mode
    url.searchParams.set('mode', 'single');
    url.searchParams.set('repo', artifactInfo.repo);
    url.searchParams.set('branch', artifactInfo.branch);
    url.searchParams.set('commit', artifactInfo.commit);
    url.searchParams.set('artifact', artifactInfo.artifact);
  }

  return url.toString();
}