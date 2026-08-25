# DO NOT MODIFY OR DELETE THIS FILE.
#
# ANY CHANGES MAY RESULT IN SERIOUS DAMAGE
# TO THE TECHNICAL INFRASTRUCTURE AND LOSS OF DATA!
#
# ACCESS TO OR USE OF PALANTIR FOUNDRY IS SUBJECT
# TO PALANTIR’S TECHNICAL SPECIFICATIONS AVAILABLE
# IN THE DOCUMENTATION. THIS WARNING CONSTITUTES AN
# ADDITION TO THOSE TECHNICAL SPECIFICATIONS AND
# NON-COMPLIANCE MAY CONSTITUTE A VIOLATION OF
# THE FOUNDRY LICENSE AGREEMENT.

import os
import requests


artifacts_uri = os.environ["ARTIFACTS_URI"]
repo_rid = os.environ["STEMMA_REPO_RID"]
password = os.environ["JOB_TOKEN"]


def raise_for_status_and_log_error(response):
    try:
        response.raise_for_status()
    except Exception as e:
        print(response.text)
        raise


def download_file(src_filename, dest_filename):
    url = (
        f"{artifacts_uri}/repositories/{repo_rid}/contents/release/files/{src_filename}"
    )
    response = requests.get(url, headers={"Authorization": f"Bearer {password}"})
    raise_for_status_and_log_error(response)

    with open(dest_filename, "wb") as f:
        f.write(response.content)


def download_plugin_with_version(version):
    ci_plugin_name = os.environ["CI_PLUGIN_NAME"]
    download_file(f"build_plugin_{version}", ci_plugin_name)
    print(f"Downloaded version {version} of the build plugin")


if __name__ == "__main__":
    build_plugin_version = os.environ.get("BUILD_PLUGIN_VERSION", "latest")
    try:
        download_plugin_with_version(build_plugin_version)
    except Exception as e:
        print(
            f"Failed to download the requested version {build_plugin_version} of the build plugin!",
            e,
        )
        raise
