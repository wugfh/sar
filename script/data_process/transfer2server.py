import subprocess

def scp_transfer(local_path, remote_host, remote_path):
    """
    使用scp将本地文件传输到远程服务器
    """
    cmd = [
        "scp",
        local_path,
        f"{remote_host}:{remote_path}"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("文件传输成功")
    except subprocess.CalledProcessError as e:
        print(f"文件传输失败: {e}")

# 示例用法
if __name__ == "__main__":
    local_file = "example.txt"
    local_preifix = "F:/sar/data/"

    remote_dir = "/home/your_username/"
    experiment_tag = 'example_16'
    local_sig_file = f'{local_preifix}{experiment_tag}_sig.mat'
    local_pos_file = f'{local_preifix}{experiment_tag}_pos.mat'
    local_param_file = f'{local_preifix}{experiment_tag}_param.mat'

    host = "host_kunlun"
    remote_path_prefix = "/home/zenghong/data/sar/data/"
    remote_sig_file = f'{remote_path_prefix}{experiment_tag}_sig.mat'
    remote_pos_file = f'{remote_path_prefix}{experiment_tag}_pos.mat'
    remote_param_file = f'{remote_path_prefix}{experiment_tag}_param.mat'
    scp_transfer(local_sig_file, host, remote_sig_file)
    scp_transfer(local_pos_file, host, remote_pos_file)
    scp_transfer(local_param_file, host, remote_param_file)