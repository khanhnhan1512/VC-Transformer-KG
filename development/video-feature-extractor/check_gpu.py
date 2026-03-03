import torch

def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("Không có GPU CUDA khả dụng.")
        return

    n = torch.cuda.device_count()
    print(f"Found {n} CUDA device(s)\n")

    for i in range(n):
        dev = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(dev)
        print(f"Device {i}: {props.name}")
        print(f"  Total memory bytes: {props.total_memory}")
        print(f"  MultiProcessor count: {props.multi_processor_count}")
        print(f"  Major.Minor: {props.major}.{props.minor}")
        print(f"  Max threads per block: {props.max_threads_per_multi_processor}")
        print("=" * 40)


if __name__ == "__main__":
    print_gpu_info()
