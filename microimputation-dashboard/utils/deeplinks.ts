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

export function createShareableUrl(baseUrl: string, artifactInfo: GitHubArtifactInfo): string {
  const url = new URL(baseUrl);

  url.searchParams.set('repo', artifactInfo.repo);
  url.searchParams.set('branch', artifactInfo.branch);
  url.searchParams.set('commit', artifactInfo.commit);
  url.searchParams.set('artifact', artifactInfo.artifact);

  return url.toString();
}