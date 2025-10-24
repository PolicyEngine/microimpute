import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const repo = searchParams.get('repo');
    const commitSha = searchParams.get('commit');

    if (!repo || !commitSha) {
        return NextResponse.json(
            { error: 'Missing repo or commit parameter' },
            { status: 400 }
        );
    }

    const githubToken = process.env.GITHUB_TOKEN;
    if (!githubToken) {
        return NextResponse.json(
            { error: 'GitHub token not configured on server' },
            { status: 500 }
        );
    }

    try {
        const [owner, repoName] = repo.split('/');

        // Get workflow runs for the commit
        const runsResponse = await fetch(
            `https://api.github.com/repos/${owner}/${repoName}/actions/runs?head_sha=${commitSha}`,
            {
                headers: {
                    Authorization: `Bearer ${githubToken}`,
                    Accept: 'application/vnd.github.v3+json',
                    'User-Agent': 'PolicyEngine-Dashboard/1.0',
                },
            }
        );

        if (!runsResponse.ok) {
            return NextResponse.json(
                { error: `GitHub API error: ${runsResponse.status}` },
                { status: runsResponse.status }
            );
        }

        const runsData = await runsResponse.json();
        const runs = runsData.workflow_runs;

        if (!runs || runs.length === 0) {
            return NextResponse.json([]);
        }

        // Collect all imputation artifacts from completed runs
        const allArtifacts = [];

        for (const run of runs) {
            if (run.status !== 'completed') continue;

            try {
                const artifactsResponse = await fetch(
                    `https://api.github.com/repos/${owner}/${repoName}/actions/runs/${run.id}/artifacts`,
                    {
                        headers: {
                            Authorization: `Bearer ${githubToken}`,
                            Accept: 'application/vnd.github.v3+json',
                            'User-Agent': 'PolicyEngine-Dashboard/1.0',
                        },
                    }
                );

                if (!artifactsResponse.ok) continue;

                const artifactsData = await artifactsResponse.json();
                const artifacts = artifactsData.artifacts;

                // Filter for imputation artifacts
                const imputationArtifacts = artifacts.filter(
                    (artifact: { name: string }) =>
                        artifact.name.toLowerCase().includes('impute') ||
                        artifact.name
                            .toLowerCase()
                            .includes('imputation') ||
                        artifact.name.toLowerCase().includes('result') ||
                        artifact.name.toLowerCase().includes('.csv')
                );

                allArtifacts.push(...imputationArtifacts);
            } catch {
                continue;
            }
        }

        // Remove duplicates and sort by creation date (newest first)
        const uniqueArtifacts = allArtifacts
            .filter(
                (artifact: { name: string }, index: number, self: Array<{ name: string }>) =>
                    index ===
                    self.findIndex((a: { name: string }) => a.name === artifact.name)
            )
            .sort(
                (a: { created_at: string }, b: { created_at: string }) =>
                    new Date(b.created_at).getTime() -
                    new Date(a.created_at).getTime()
            );

        return NextResponse.json(uniqueArtifacts);
    } catch (error) {
        return NextResponse.json(
            {
                error:
                    error instanceof Error
                        ? error.message
                        : 'Unknown error',
            },
            { status: 500 }
        );
    }
}
