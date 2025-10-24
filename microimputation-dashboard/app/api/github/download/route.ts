import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const url = searchParams.get('url');

    if (!url) {
        return NextResponse.json(
            { error: 'Missing url parameter' },
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
        const downloadResponse = await fetch(url, {
            headers: {
                Authorization: `Bearer ${githubToken}`,
                Accept: 'application/vnd.github.v3+json',
                'User-Agent': 'PolicyEngine-Dashboard/1.0',
            },
        });

        if (!downloadResponse.ok) {
            return NextResponse.json(
                { error: `GitHub API error: ${downloadResponse.status}` },
                { status: downloadResponse.status }
            );
        }

        // Get the artifact ZIP as an ArrayBuffer
        const zipBuffer = await downloadResponse.arrayBuffer();

        // Return the ZIP file as a response
        return new NextResponse(zipBuffer, {
            headers: {
                'Content-Type': 'application/zip',
                'Content-Length': zipBuffer.byteLength.toString(),
            },
        });
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
