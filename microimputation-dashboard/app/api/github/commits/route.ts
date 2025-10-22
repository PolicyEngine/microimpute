import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const repo = searchParams.get('repo');
    const branch = searchParams.get('branch');

    if (!repo || !branch) {
        return NextResponse.json(
            { error: 'Missing repo or branch parameter' },
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
        const response = await fetch(
            `https://api.github.com/repos/${repo}/commits?sha=${branch}&per_page=20`,
            {
                headers: {
                    Authorization: `Bearer ${githubToken}`,
                    Accept: 'application/vnd.github.v3+json',
                    'User-Agent': 'PolicyEngine-Dashboard/1.0',
                },
            }
        );

        if (!response.ok) {
            return NextResponse.json(
                { error: `GitHub API error: ${response.status}` },
                { status: response.status }
            );
        }

        const commits = await response.json();
        return NextResponse.json(commits);
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
