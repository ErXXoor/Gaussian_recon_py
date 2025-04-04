import subprocess
import os


def run_rvd(geo_path, input_path, output_path):
    params = [
        "",
        "",
        "true",
    ]

    params[0] = input_path
    params[1] = output_path

    command = [geo_path] + params
    print(command)

    try:
        return_code = subprocess.call(command)
        print("Return code:", return_code)
    except Exception as e:
        print("An error occurred:", str(e))


def run_rvd_hd(geo_path, input_path, output_path):
    params = [
        "",
        "",
        "6",
        "10000",
    ]

    params[0] = input_path
    params[1] = output_path

    command = [geo_path] + params
    print(command)

    try:
        return_code = subprocess.call(command)
        print("Return code:", return_code)
    except Exception as e:
        print("An error occurred:", str(e))
