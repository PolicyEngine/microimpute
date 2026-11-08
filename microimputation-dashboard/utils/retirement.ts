// PolicyEngine archived microimpute, so the dashboard no longer loads results
// from GitHub. The four GitHub API routes answer 410 with this message, and
// the upload page shows it to anyone who opens a link the old GitHub loader
// shared.
export const GITHUB_LOADER_RETIRED =
  'PolicyEngine archived microimpute, so this dashboard no longer loads results from GitHub. Upload a results CSV instead.';

// The old loader's share links named a GitHub Actions artifact with these
// four query parameters.
const SHARE_LINK_PARAMS = ['repo', 'branch', 'commit', 'artifact'];

export function isGitHubShareLink(searchParams: URLSearchParams): boolean {
  return SHARE_LINK_PARAMS.every(name => Boolean(searchParams.get(name)));
}
