import ast

class LinearCodeGenerator(ast.NodeVisitor):
    def __init__(self):
        self.linear_code = []

    def generate_linear_code(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return '\n'.join(self.linear_code)

    def generic_visit(self, node):
        self.linear_code.append(ast.dump(node))
        ast.NodeVisitor.generic_visit(self, node)

# 示例代码
code = """
def square(x):
    return x * x

result = square(5)
"""

# 生成线性表示
generator = LinearCodeGenerator()
linear_code = generator.generate_linear_code(code)

# 打印线性表示
print("Linear Code:")
print(linear_code)