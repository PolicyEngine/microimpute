import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const repo = searchParams.get('repo');

    if (!repo) {
        return NextResponse.json(
            { error: 'Missing repo parameter' },
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
        const allBranches = [];
        let page = 1;
        const perPage = 100;

        while (true) {
            const response = await fetch(
                `https://api.github.com/repos/${repo}/branches?per_page=${perPage}&page=${page}`,
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

            const branches = await response.json();

            if (branches.length === 0) {
                break;
            }

            allBranches.push(...branches);

            if (branches.length < perPage) {
                break;
            }

            page++;

            if (page > 10) {
                break;
            }
        }

        return NextResponse.json(allBranches);
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
