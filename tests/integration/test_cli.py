import subprocess


def test_cli_help():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "CLI for running AIDE workflows" in result.stdout


def test_cli_train_help():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Run an AIDE experiment" in result.stdout


def test_cli_init_help():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "init", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Create a new AIDE experiment scaffold in the target directory" in result.stdout


def test_cli_list_help():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "list", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert (
        "List registered models, components, or transforms for the current project" in result.stdout
    )


def test_cli_train_no_args():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "train"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: --experiment" in result.stderr


def test_cli_init_no_args():
    # Checks that the CLI entrypoint installed properly in Docker
    result = subprocess.run(["aide", "init"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: <target_dir>" in result.stderr
