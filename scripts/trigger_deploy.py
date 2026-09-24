"""Ask GitHub Actions to run deploy.yml (workflow_dispatch) for a promoted model version.

Needs a GitHub token with Actions: write on the repo (fine-grained PAT).
Usage: GITHUB_REPO=owner/repo GITHUB_TOKEN=... python -m scripts.trigger_deploy --model-version 3
"""
import argparse
import os
import sys

import requests

WORKFLOW_FILE = "deploy.yml"


def trigger_deploy(model_version: str, repo: str, token: str, ref: str = "main") -> bool:
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": {"model_version": str(model_version)}},
        timeout=10,
    )
    # GitHub answers 204 (or 200 when returning run details) on success.
    if response.status_code not in (200, 204):
        print(f"deploy trigger failed: {response.status_code} {response.text}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    ok = trigger_deploy(args.model_version, os.environ["GITHUB_REPO"], os.environ["GITHUB_TOKEN"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
