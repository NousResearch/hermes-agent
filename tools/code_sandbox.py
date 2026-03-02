import subprocess
import sys
import os
import tempfile

def execute_python_code(code_snippet: str):
    """
    Executes a Python code snippet in a secure subprocess and captures the output or errors.
    This allows Hermes to verify if its generated code actually works before presenting it.
    """
    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file.write(code_snippet.encode('utf-8'))
        temp_path = temp_file.name

    try:
        # Run the code and capture output
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=10  # Safety timeout
        )
        
        # Cleanup the file
        os.remove(temp_path)
        
        if result.returncode == 0:
            return f"✅ EXECUTION SUCCESSFUL:\n{result.stdout}"
        else:
            return f"❌ EXECUTION FAILED:\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        os.remove(temp_path)
        return "⚠️ ERROR: Code execution timed out (Infinite loop suspected)."
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"⚠️ SYSTEM ERROR: {str(e)}"
