Here is a markdown file with examples of commonly used Linux commands to accomplish the tasks you mentioned:

```markdown
# Using SSH on Windows to Access a Linux Server and Manage Files and Run PyTorch Code

## Connecting to the Linux Server via SSH

1. Open the Command Prompt or PowerShell on your Windows machine.
2. Use the `ssh` command to connect to the Linux server:
   ```
   ssh username@server_ip_address
   ```
   Replace `username` with your Linux server username and `server_ip_address` with the IP address or hostname of the server.
3. Enter your password when prompted to authenticate the connection.

## Uploading and Downloading Files

1. To upload a file from your Windows machine to the Linux server, use the `scp` command:
   ```
   scp /path/to/local/file username@server_ip_address:/path/to/remote/directory
   ```
   Replace `/path/to/local/file` with the path to the file on your Windows machine, `username` and `server_ip_address` with your server details, and `/path/to/remote/directory` with the destination directory on the Linux server.

2. To download a file from the Linux server to your Windows machine, use the `scp` command in reverse order:
   ```
   scp username@server_ip_address:/path/to/remote/file /path/to/local/directory
   ```
   Replace `/path/to/remote/file` with the path to the file on the Linux server, `username` and `server_ip_address` with your server details, and `/path/to/local/directory` with the destination directory on your Windows machine.

## Managing Folders and Files on the Linux Server

1. To navigate between directories, use the `cd` command:
   ```
   cd /path/to/directory
   ```
   Replace `/path/to/directory` with the desired directory path.

2. To list files and directories in the current directory, use the `ls` command:
   ```
   ls
   ```

3. To create a new directory, use the `mkdir` command:
   ```
   mkdir directory_name
   ```
   Replace `directory_name` with the desired name for the new directory.

4. To remove a file, use the `rm` command:
   ```
   rm file_name
   ```
   Replace `file_name` with the name of the file you want to remove.

5. To remove a directory and its contents, use the `rm` command with the `-r` flag:
   ```
   rm -r directory_name
   ```
   Replace `directory_name` with the name of the directory you want to remove.

## Running PyTorch Code on GPUs

1. Ensure that PyTorch is installed on the Linux server with GPU support.

2. Connect to the Linux server via SSH.

3. Navigate to the directory where your PyTorch code is located using the `cd` command.

4. Run your PyTorch code using the Python interpreter:
   ```
   python your_pytorch_script.py
   ```
   Replace `your_pytorch_script.py` with the name of your PyTorch script file.

5. If you want to specify which GPU to use for running the code, you can set the `CUDA_VISIBLE_DEVICES` environment variable:
   ```
   CUDA_VISIBLE_DEVICES=0 python your_pytorch_script.py
   ```
   Replace `0` with the desired GPU index (e.g., 1, 2, etc.) if you have multiple GPUs available.

These are some of the commonly used Linux commands for accessing a server via SSH, managing files, and running PyTorch code on GPUs. Remember to replace the placeholders (e.g., `username`, `server_ip_address`, file paths) with your actual server details and file paths.
```

This markdown file provides examples of SSH connection, file upload/download using `scp`, basic file and directory management commands, and instructions for running PyTorch code on GPUs on the Linux server.
