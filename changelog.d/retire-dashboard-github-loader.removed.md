Retire the dashboard's GitHub loader: its API routes no longer read the server's GITHUB_TOKEN, and the download route no longer forwards that token to whatever URL a caller names.
