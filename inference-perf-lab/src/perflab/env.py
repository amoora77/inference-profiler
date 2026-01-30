import os
import platform
import sys
import torch


def get_env_info():
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info
