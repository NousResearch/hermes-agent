import ast
import os
import sys

def scan_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        return
        
    for node in ast.walk(tree):
        # 1. Silent Failures: except blocks that just pass
        if isinstance(node, ast.ExceptHandler):
            # check if body contains only pass
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                print(f"[Silent Failure] {filepath}:{node.lineno}: empty except block (pass)")
            # also check if body has no logging (rudimentary check: no 'log', 'print', 'warning', 'error', 'debug' in body)
            else:
                has_log = False
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                        if stmt.func.attr in ('info', 'warning', 'error', 'exception', 'debug', 'print'):
                            has_log = True
                    elif isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name):
                        if stmt.func.id in ('print', 'logger'):
                            has_log = True
                if not has_log and not any(isinstance(stmt, ast.Raise) for stmt in ast.walk(node)) and not any(isinstance(stmt, ast.Return) for stmt in ast.walk(node)) and not any(isinstance(stmt, ast.Continue) for stmt in ast.walk(node)):
                    # Check if there's any external call that could be considered handling, otherwise flag
                    pass
        
        # 3. Resource Leaks: open() without with
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'open':
                print(f"[Resource Leak] {filepath}:{node.lineno}: open() called outside of 'with' block (assigned to variable)")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'open':
                # Check if this call is part of a With context
                # AST parent traversing isn't directly available without custom passing, but we do simplistic checks.
                pass

        # 2. Async Issues: subprocess in async without timeout?
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ('create_subprocess_shell', 'create_subprocess_exec', 'Popen', 'run'):
                    print(f"[Async/Subprocess] {filepath}:{node.lineno}: subprocess call found, verify error handling and timeout.")

def main():
    dirs_to_scan = ['agent', 'core', 'tools', 'gateway', 'hermes_cli']
    for d in dirs_to_scan:
        if not os.path.isdir(d):
            if d == 'core':
                print(f"'core' directory not found.")
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith('.py'):
                    scan_file(os.path.join(root, f))
    
    # Also scan root py files like run_agent.py
    for f in os.listdir('.'):
        if f.endswith('.py') and os.path.isfile(f):
            scan_file(f)

if __name__ == '__main__':
    main()
