import time
import torch
import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from dual_network import DualNetwork

BOARDSIZE  = 7
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]
WARMUP     = 100
REPS       = 500

configs = [
    ("64f / 6res",   64,  6),
    ("128f / 10res", 128, 10),
]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = None

print(f"CPU threads : {torch.get_num_threads()}")
print(f"GPU         : {gpu_name or 'not available'}")
print()

def bench(model, device, batch_size, warmup, reps):
    model = model.to(device)
    model.eval()
    x = torch.randn(batch_size, 8, BOARDSIZE, BOARDSIZE, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed loop
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ms_per_call = elapsed / reps * 1000
    us_per_pos  = elapsed / reps / batch_size * 1e6
    return ms_per_call, us_per_pos


for label, filters, res in configs:
    params = sum(p.numel() for p in DualNetwork(boardsize=BOARDSIZE, filters=filters, num_residual=res).parameters())
    print(f"{'='*66}")
    print(f"{label}  ({params:,} params)")
    print(f"{'='*66}")

    # Header
    device_labels = ["CPU"] + ([gpu_name] if gpu_name else [])
    col = max(len(d) for d in device_labels) + 2

    header = f"  {'batch':>6}"
    for d in device_labels:
        header += f"  {d+' ms/call':>{col+8}}  {'µs/pos':>8}"
    if len(devices) > 1:
        header += f"  {'speedup':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for bs in BATCH_SIZES:
        model = DualNetwork(boardsize=BOARDSIZE, filters=filters, num_residual=res)
        results = []
        for device in devices:
            ms, us = bench(model, device, bs, WARMUP, REPS)
            results.append((ms, us))

        row = f"  {bs:>6}"
        for ms, us in results:
            row += f"  {ms:>{col+2}.3f} ms/call  {us:>6.1f} µs"
        if len(results) == 2:
            speedup = results[0][0] / results[1][0]
            row += f"  {speedup:>6.1f}×"
        print(row)

    print()

# Extra: measure host→device transfer overhead for a single position (bs=1)
if gpu_name:
    print(f"{'='*66}")
    print("Host → device transfer overhead  (bs=1, single float32 tensor)")
    print(f"{'='*66}")
    x_cpu = torch.randn(1, 8, BOARDSIZE, BOARDSIZE)
    # warmup
    for _ in range(200):
        x_cpu.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000):
        x_cpu.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    transfer_us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"  Blocking .to('cuda'): {transfer_us:.1f} µs/call")
    # non-blocking
    for _ in range(200):
        x_cpu.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000):
        x_cpu.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    transfer_nb_us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"  Non-blocking .to('cuda'): {transfer_nb_us:.1f} µs/call")
    print(f"\n  (compare against ~{transfer_us:.0f} µs overhead per MCTS simulation")
    print(f"   if each sim calls the GPU with a single position)")
