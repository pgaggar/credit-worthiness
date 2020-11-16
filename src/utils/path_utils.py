from pathlib import Path

def get_app_data_path() -> Path:
    '''
    method to provide app root directory directory path i.e */credit-worthiness/
    :return: app root directory directory path
    '''
    root_path = get_app_src_path().parent.joinpath('data')
    return root_path


def get_app_src_path() -> Path:
    '''
     method to provide "src" directory path i.e */credit-worthiness/src/.
     path is derived from this "utils.py" file current location
     :return: "src" directory path
    '''
    src_path = Path(__file__).parent.parent
    return src_path

def get_app_resources_path() -> Path:
    '''
     method to provide "resources" directory path i.e */credit-worthiness/src/resources.
     path is derived from this "utils.py" file current location
     :return: "src" directory path
    '''
    src_path = get_app_src_path().joinpath('resources')
    return src_path