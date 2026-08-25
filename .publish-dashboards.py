# DO NOT MODIFY OR DELETE THIS FILE.
import json
import os
import re
import tarfile
from urllib import request


def get_environment_variable(variable):
    if variable not in os.environ:
        raise RuntimeError("'%s' environment variable was not found." % variable)
    if os.environ.get(variable) == "https://ci-role-not-found":
        raise RuntimeError("'%s' role was not found." % variable)
    return os.environ.get(variable)


UTF_8 = "utf-8"

FOUNDRY_ARTIFACTS_API = get_environment_variable("ARTIFACTS_URI")
FOUNDRY_CONTAINER_SERVICE_API = get_environment_variable("FOUNDRY_CONTAINER_SERVICE_API")
STEMMA_API = get_environment_variable("STEMMA_API")

BRANCH = get_environment_variable("JEMMA_BRANCH")
REPOSITORY_RID = get_environment_variable("REPOSITORY_RID")
TOKEN = get_environment_variable("JOB_TOKEN")
COMMITISH = get_environment_variable("GIT_COMMIT_ID")
VERSION = get_environment_variable("GIT_VERSION")

ARCHIVE_FILE = "archive.tgz"
MAVEN_COORDINATE = "{0}:archive:{1}".format(REPOSITORY_RID, VERSION)
MAVEN_LOCATOR = "{0}/archive/{1}/archive-{1}.tgz".format(REPOSITORY_RID.replace(".", "/"), VERSION)


def print_and_raise(e):
    print("Encountered HTTP error.")
    print("HTTP Status code: {0}".format(e.getcode()))
    print("Reason: {0}".format(e.reason))
    print("Response: {0}".format(e.read()))
    raise RuntimeError("Failed to publish dashboard.", e)


def publish_code():
    with tarfile.open(ARCHIVE_FILE, "w:gz") as tar:
        for name in os.listdir("."):
            tar.add(name)
    publish_url = "{0}/repositories/{1}/contents/release/maven/{2}".format(FOUNDRY_ARTIFACTS_API, REPOSITORY_RID, MAVEN_LOCATOR)

    with open(ARCHIVE_FILE, "rb") as data:
        req = request.Request(publish_url, data)
        req.get_method = lambda: "PUT"
        req.add_header("Authorization", "Bearer {0}".format(TOKEN))
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Accept", "*/*")
        try:
            request.urlopen(req)
        except request.HTTPError as e:
            if (e.getcode() == 409):
                # Ignore conflicts, it simply means the archive has already been uploaded,
                # e.g. when retriggering checks or checking out a new branch
                return
            print_and_raise(e)


def get_fallback_branches():
    url = "{0}/repos/{1}/fallbacks".format(STEMMA_API, REPOSITORY_RID)
    try:
        req = request.Request(url)
        req.get_method = lambda: "GET"
        req.add_header("Authorization", "Bearer {0}".format(TOKEN))
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "*/*")
        response = request.urlopen(req)
        return json.loads(response.read().decode(UTF_8)).get("refs/heads/{0}".format(BRANCH), {'names': []})['names']
    except request.HTTPError as e:
        print_and_raise(e)


def publish_dashboard(dashboard_rid, dashboard_definition):
    publish_url = "{0}/containers/{1}/versions".format(FOUNDRY_CONTAINER_SERVICE_API, dashboard_rid)
    runtime = {
        key: dashboard_definition.get(key)
        for key in ["idleTimeout", "resourceProfile", "resources", "resourcesV2", "inputs", "networkPolicies", "sources", "environmentVariables", "shared", "allowDownloads"]
        if key in dashboard_definition
    }
    # FCS will only apply resourcesV2 if useResourcesV2 is set to true
    if "resourcesV2" in runtime:
        runtime["resourcesV2"]["useResourcesV2"]=True
    payload = {
        "branch": BRANCH,
        "fallbackBranches": get_fallback_branches(),
        "runtime": {
            **runtime,
            "mounts": {
                "repo": MAVEN_COORDINATE
            },
            "sourceProvenance": {
                "repositoryRid": REPOSITORY_RID,
                "commitish": COMMITISH,
                "file": ".dashboards/{0}.json".format(dashboard_rid)
            }
        }
    }

    try:
        req = request.Request(publish_url, json.dumps(payload).encode(UTF_8))
        req.get_method = lambda: "POST"
        req.add_header("Authorization", "Bearer {0}".format(TOKEN))
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "*/*")
        request.urlopen(req)
    except request.HTTPError as e:
        print_and_raise(e)


def publish_dashboards():
    dashboards_dir = ".dashboards"
    if os.path.isdir(dashboards_dir):
        for name in os.listdir(dashboards_dir):
            if not re.match(r"ri\..+\.json", name):
                continue
            with open(os.path.join(dashboards_dir, name), "r") as in_f:
                dashboard_definition = json.load(in_f)
            dashboard_rid = name[:-len(".json")]
            publish_dashboard(dashboard_rid, dashboard_definition)
            print("Published new version of dashboard {0}".format(dashboard_rid))


publish_code()
publish_dashboards()
