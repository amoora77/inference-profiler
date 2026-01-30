import numpy as np


def compute_percentiles(values, percentiles=[50, 90, 95, 99]):
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}
    arr = np.array(values)
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(arr, p))
    return result


def compute_throughput(total_requests, total_time_ms):
    if total_time_ms <= 0:
        return 0.0
    return (total_requests / total_time_ms) * 1000


def compute_samples_per_sec(total_samples, total_time_ms):
    if total_time_ms <= 0:
        return 0.0
    return (total_samples / total_time_ms) * 1000


def get_peak_rss_mb():
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if hasattr(rusage, "ru_maxrss"):
            if resource.RLIMIT_RSS == 0:
                return rusage.ru_maxrss / (1024 * 1024)
            return rusage.ru_maxrss / 1024
        return 0.0
    except Exception:
        return 0.0
