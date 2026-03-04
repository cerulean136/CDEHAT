import subprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def execute_command(in_command):
    print(f"Current working directory: {os.getcwd()}")
    in_command = ' '.join(in_command) if isinstance(in_command, list) else in_command
    print(f"Preparing to execute command: {in_command}")
    with subprocess.Popen(args=in_command, stdout=subprocess.PIPE, text=True, bufsize=1,
                          universal_newlines=True) as proc:
        for line in proc.stdout:
            print(line, end='')
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd=in_command)
