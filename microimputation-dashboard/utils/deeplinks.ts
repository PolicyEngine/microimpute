// Deeplink utilities for GitHub artifact sharing

export interface GitHubArtifactInfo {
  repo: string;
  branch: string;
  commit: string;
  artifact: string;
}

export interface DeeplinkParams {
  primary?: GitHubArtifactInfo;
}

export function parseDeeplinkParams(searchParams: URLSearchParams): DeeplinkParams | null {
  const repo = searchParams.get('repo');
  const branch = searchParams.get('branch');
  const commit = searchParams.get('commit');
  const artifact = searchParams.get('artifact');

  if (!repo || !branch || !commit || !artifact) {
    return null;
  }

  return {
    primary: {
      repo,
      branch,
      commit,
      artifact,
    },
  };
}

export function createShareableUrl(artifactInfo: GitHubArtifactInfo): string {
  const baseUrl = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.host}${window.location.pathname}`
    : '';

  const urlParams = new URLSearchParams();
  urlParams.set('repo', artifactInfo.repo);
  urlParams.set('branch', artifactInfo.branch);
  urlParams.set('commit', artifactInfo.commit);
  urlParams.set('artifact', artifactInfo.artifact);

  return `${baseUrl}?${urlParams.toString()}`;
}