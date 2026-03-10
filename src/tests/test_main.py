from main import my_app
from hydra import compose, initialize


def test_my_app():
    """
    Test the main application functionality using a dummy configuration.

    This test initializes the application with a specified configuration path
    and job name. It modifies the logger settings to disable offline logging
    and model logging, and sets the callbacks to None. The main application
    function is then called with the modified configuration to verify that
    the training loop operates correctly.

    Dependencies:
    - `initialize`: Function to set up the application environment.
    - `compose`: Function to create a configuration object from the specified
        configuration name and overrides.
    - `my_app`: The main application function to be tested.

    """
    with initialize(version_base=None, config_path="../../configs", job_name="config"):
        # check if the main training loop is working with a dummy config
        cfg = compose(config_name="config", overrides=["experiment=dummy"])
        cfg["logger"]["offline"] = True
        cfg["logger"]["log_model"] = False
        cfg["callbacks"] = None
        cfg["trainer"]["fast_dev_run"] = True  # run a single batch for testing
        my_app(cfg)


if __name__ == "__main__":
    test_my_app()
