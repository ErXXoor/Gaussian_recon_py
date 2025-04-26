import numpy as np
import fpsample


def parse_id_list(file_path):
    id_list = []
    with open(file_path, 'r') as f:
        for line in f:
            obj_id = line.strip()
            id_list.append(obj_id)
    return id_list


if __name__ == "__main__":
    id_path = "/home/hongbo/Desktop/code/PTV3_Embedding/obj_lists/test_80.txt"
    id_list = parse_id_list(id_path)

    root_path = ""

    for obj_id in id_list:
        xyz_path =
