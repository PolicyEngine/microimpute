import { NextResponse } from 'next/server';
import { GITHUB_LOADER_RETIRED } from '@/utils/retirement';

// These routes used to call GitHub with the server's GITHUB_TOKEN for any
// caller, and the download route sent that token to whatever URL the caller
// named. They now read no token and answer 410 Gone.
export function retiredGitHubRoute() {
    return NextResponse.json({ error: GITHUB_LOADER_RETIRED }, { status: 410 });
}
