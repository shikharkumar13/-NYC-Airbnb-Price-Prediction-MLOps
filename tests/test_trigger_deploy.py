import pytest

from scripts import trigger_deploy as td


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_post(url, headers, json, timeout):
        calls.update(url=url, headers=headers, json=json)
        return calls.get("response", FakeResponse(204))

    monkeypatch.setattr(td.requests, "post", fake_post)
    return calls


def test_trigger_deploy_dispatches_deploy_workflow(captured):
    assert td.trigger_deploy("3", repo="me/airbnb", token="tok") is True
    assert captured["url"] == "https://api.github.com/repos/me/airbnb/actions/workflows/deploy.yml/dispatches"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["json"] == {"ref": "main", "inputs": {"model_version": "3"}}


def test_trigger_deploy_accepts_200_with_run_details(captured):
    captured["response"] = FakeResponse(200)
    assert td.trigger_deploy("3", repo="me/airbnb", token="tok") is True


def test_trigger_deploy_raises_with_githubs_reason(captured):
    # A rejected trigger must not look like success: the Prefect task has to
    # fail, and GitHub's reason has to reach the Prefect logs.
    captured["response"] = FakeResponse(403, "Resource not accessible by personal access token")
    with pytest.raises(td.DeployTriggerError, match="403.*Resource not accessible"):
        td.trigger_deploy("3", repo="me/airbnb", token="weak")


def test_request_deploy_task_fails_when_github_rejects(captured, monkeypatch):
    import orchestrate_training

    monkeypatch.setenv("GITHUB_REPO", "me/airbnb")
    monkeypatch.setenv("GITHUB_TOKEN", "weak")
    captured["response"] = FakeResponse(403, "Resource not accessible by personal access token")
    with pytest.raises(td.DeployTriggerError):
        orchestrate_training.request_deploy.fn("4")
