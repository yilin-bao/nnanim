import os
import re

# def extract_functions_from_file(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()

#     pattern = re.compile(r"class\s+(\w+)\s*\(")
#     matches = pattern.findall(content)

#     return matches


# def load_functions_from_folder(folder_path):
#     functions = []

#     files = [f for f in os.listdir(folder_path) if f.endswith('.py')]

#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         file_functions = extract_functions_from_file(file_path)
#         functions.extend(file_functions)

#     return functions

def folder_scanner(directory):
    file_list = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def file_selector(file_list):
    ret = []
    for file in files:
        if file.endswith(".py"):
            ret.append(file)
    return ret


def single_file_scanner(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


# directory_to_scan = "/Users/zzyang/Documents/NNanim/TestingCode/"  # 替换成实际的目录路径
# files = folder_scanner(directory_to_scan)
# files = file_selector(files)
# print("Files in the directory:")
# for file in files:
#     print("="*20, file, "="*20)
#     lines = single_file_scanner(file)
#     for line in lines:
#         print(line)