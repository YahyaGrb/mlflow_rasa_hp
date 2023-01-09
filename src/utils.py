
import json
from urllib.parse import urlparse
import os

def _transform_uri_to_path(uri, sub_directory = ""):
    """transform run returned artifacts uri to path

    Args:
        uri (uri): uri returned by a run
        sub_directory (str, optional): sub directory to select artifact from if necessary. Defaults to "".

    Returns:
        path: path of the artifact returned
    """
    parsed_url = urlparse(uri)
    # Get the path from the parsed URL
    path = parsed_url.path
    # Make the path absolute using os.path.abspath()
    abs_path = os.path.abspath(path)
    # Get the list of files in the directory
    if sub_directory:
        files_path = os.path.join(abs_path, sub_directory)
        files = os.listdir(files_path)
        # Get the first file in the list (assuming there is at least one file in the directory)
        file = files[0]
        # Create the new path by combining the file name with the current working directory
        return os.path.join(files_path, file)
    return abs_path

def _get_run_config(template_path, params, destination):
    """generates a configuration file based on the given template and parameters.

    Args:
        template_path (str): path to template config
        params (dict): key:value pari dict for params
        destination (str): destination file
    Returns:
        file_path (str): path to generated configuration file
    """
    file_path = os.path.join(destination, "run_config.yml")
    if isinstance(params,(str)):
        params = json.loads(params)
    with open(template_path) as f:
        run_config_yml = f.read().format(**params)
        with open(file_path, 'w+') as temp_f:
            temp_f.write(run_config_yml)
    return file_path