import { NextResponse } from 'next/server';

// PolicyEngine archived microimpute, so the dashboard no longer loads results
// from GitHub. These routes used to call GitHub with the server's GITHUB_TOKEN
// for any caller, and the download route sent that token to whatever URL the
// caller named. They now read no token and answer 410 Gone.
export function retiredGitHubRoute() {
    return NextResponse.json(
        {
            error:
                'PolicyEngine archived microimpute, so this dashboard no longer loads results from GitHub. Upload a results CSV instead.',
        },
        { status: 410 }
    );
}
