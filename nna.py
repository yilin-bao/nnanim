import ast
import inspect
import os

def find_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

directory_path = "/Users/zzyang/Documents/NNanim/TestingCode"
python_files = find_python_files(directory_path)

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# 获取神经网络类的源代码
# network_source = inspect.getsource(YourNetworkClass)
network_source = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/modules.py")
# 解析神经网络类的 AST
tree = ast.parse(network_source)

class NetworkAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.layers = []

    def visit_ClassDef(self, node):
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                self.visit_FunctionDef(member)

    def visit_FunctionDef(self, node):
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                layer_info = {
                    "name": statement.value.func.attr,
                    "parameters": [],
                }
                for arg in statement.value.args:
                    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "self":
                        layer_info["parameters"].append(arg.attr)
                self.layers.append(layer_info)

# 创建 NetworkAnalyzer 实例并访问 AST
network_analyzer = NetworkAnalyzer()
network_analyzer.visit(tree)

# 打印每一层的结构信息
for layer in network_analyzer.layers:
    print(layer)