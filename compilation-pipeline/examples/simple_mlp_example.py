# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple MLP Example for End-to-End Compilation Tracing

This script demonstrates the complete compilation and execution pipeline
from PyTorch to Tenstorrent hardware. It corresponds to the example in
../09-end-to-end-example.md

Usage:
    # Basic execution
    python simple_mlp_example.py

    # Dump intermediate IRs
    python simple_mlp_example.py --dump-ir

    # Enable profiling
    python simple_mlp_example.py --profile

    # Compare with CPU reference
    python simple_mlp_example.py --verify
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Two-layer MLP for demonstration.

    Architecture:
        Input: [batch_size, 128]
        Hidden Layer: Linear(128, 256) + ReLU
        Output Layer: Linear(256, 10)
        Output: [batch_size, 10]
    """

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Layer 1: Linear + ReLU
        x = self.fc1(x)  # [batch, 128] @ [128, 256] + [256] -> [batch, 256]
        x = self.relu(x)  # [batch, 256] -> [batch, 256]

        # Layer 2: Linear
        x = self.fc2(x)  # [batch, 256] @ [256, 10] + [10] -> [batch, 10]
        return x


def setup_environment(args):
    """Configure environment variables for debugging/profiling."""
    if args.dump_ir:
        os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
        os.environ["TT_MLIR_DUMP_IR"] = "1"

        ir_dump_path = args.ir_path or "/tmp/mlp_ir_dump"
        os.environ["TT_MLIR_DUMP_IR_PATH"] = ir_dump_path
        print(f"IR dumps will be saved to: {ir_dump_path}")

    if args.profile:
        os.environ["TT_METAL_PROFILER"] = "1"

        profile_path = args.profile_path or "/tmp/mlp_profile.json"
        os.environ["TT_METAL_PROFILER_OUTPUT"] = profile_path
        print(f"Profile will be saved to: {profile_path}")

    if args.verbose:
        os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
        os.environ["TT_METAL_LOGGER_LEVEL"] = "DEBUG"


def verify_device():
    """Verify TT device is available."""
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print(f"✓ TT device found: {device}")
        return True
    except Exception as e:
        print(f"✗ Failed to find TT device: {e}")
        print("\nTroubleshooting:")
        print("  1. Check device status: tt-smi")
        print("  2. Reset device if hung: tt-smi --reset 0")
        print("  3. Verify TTMLIR_TOOLCHAIN_DIR is set")
        return False


def run_inference_tt(model, input_tensor, batch_size=32, warmup=1, iterations=10):
    """Run inference on TT device with timing.

    Args:
        model: Compiled model
        input_tensor: Input tensor on TT device
        batch_size: Batch size for input
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Tuple of (output_tensor, avg_time_ms)
    """
    import torch_xla.core.xla_model as xm

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        output = model(input_tensor)
        xm.mark_step()
        if i == 0:
            print("  First inference (triggers compilation)...")
        else:
            print(f"  Warmup iteration {i + 1}/{warmup}")

    # Timed iterations
    print(f"\nRunning {iterations} timed iterations...")
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        output = model(input_tensor)
        xm.mark_step()
        end = time.perf_counter()

        iteration_time = (end - start) * 1000  # Convert to ms
        times.append(iteration_time)
        print(f"  Iteration {i + 1}/{iterations}: {iteration_time:.2f} ms")

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return output, avg_time, std_time


def run_inference_cpu(model, input_tensor, iterations=10):
    """Run inference on CPU with timing.

    Args:
        model: CPU model
        input_tensor: Input tensor on CPU
        iterations: Number of timed iterations

    Returns:
        Tuple of (output_tensor, avg_time_ms)
    """
    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()

            times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return output, avg_time, std_time


def verify_correctness(output_tt, output_cpu, tolerance=1e-3):
    """Compare TT output with CPU reference.

    Args:
        output_tt: Output from TT device
        output_cpu: Output from CPU
        tolerance: Maximum acceptable difference

    Returns:
        True if outputs match within tolerance
    """
    print("\nVerifying correctness...")

    # Move TT output to CPU for comparison
    output_tt_cpu = output_tt.cpu()

    # Compute differences
    abs_diff = torch.abs(output_tt_cpu - output_cpu)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Tolerance: {tolerance:.6f}")

    if max_diff <= tolerance:
        print("  ✓ Outputs match within tolerance")
        return True
    else:
        print("  ✗ Outputs differ more than tolerance")
        print(f"\n  Sample differences:")
        print(f"    TT:  {output_tt_cpu[0, :5]}")
        print(f"    CPU: {output_cpu[0, :5]}")
        print(f"    Diff: {abs_diff[0, :5]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple MLP example for TT compilation pipeline"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--dump-ir", action="store_true", help="Dump intermediate IR representations"
    )
    parser.add_argument(
        "--ir-path", type=str, help="Path to save IR dumps (default: /tmp/mlp_ir_dump)"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        help="Path to save profile (default: /tmp/mlp_profile.json)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify results against CPU reference",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup iterations (default: 1)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timed iterations (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Simple MLP Example - TT Compilation Pipeline")
    print("=" * 70)

    # Setup environment
    setup_environment(args)

    # Verify TT device
    if not verify_device():
        return 1

    # Import after device verification
    import torch_xla.core.xla_model as xm

    # Create model
    print("\n" + "─" * 70)
    print("Creating model...")
    print("─" * 70)
    model = SimpleMLP()
    print(f"Model architecture:\n{model}")

    # Move to TT device
    print("\n" + "─" * 70)
    print("Moving model to TT device...")
    print("─" * 70)
    device = xm.xla_device()
    model = model.to(device)

    # Compile with torch.compile
    print("\n" + "─" * 70)
    print("Compiling model with torch.compile(backend='tt')...")
    print("─" * 70)
    compiled_model = torch.compile(model, backend="tt")

    # Create input tensor
    print("\n" + "─" * 70)
    print(f"Creating input tensor (batch_size={args.batch_size})...")
    print("─" * 70)
    input_tensor = torch.randn(args.batch_size, 128, device=device)
    print(f"Input shape: {input_tensor.shape}")

    # Run inference on TT
    print("\n" + "─" * 70)
    print("Running inference on TT device...")
    print("─" * 70)
    output_tt, avg_time_tt, std_time_tt = run_inference_tt(
        compiled_model,
        input_tensor,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print(f"\n✓ TT Inference: {avg_time_tt:.2f} ± {std_time_tt:.2f} ms")
    print(f"  Output shape: {output_tt.shape}")
    print(f"  Output sample: {output_tt[0, :5]}")

    # Verify against CPU if requested
    if args.verify:
        print("\n" + "─" * 70)
        print("Verifying against CPU reference...")
        print("─" * 70)

        # Create CPU model with same weights
        model_cpu = SimpleMLP()
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()

        # Create CPU input
        input_cpu = input_tensor.cpu()

        # Run inference on CPU
        output_cpu, avg_time_cpu, std_time_cpu = run_inference_cpu(
            model_cpu, input_cpu, iterations=args.iterations
        )

        print(f"\n✓ CPU Inference: {avg_time_cpu:.2f} ± {std_time_cpu:.2f} ms")

        # Compare results
        verify_correctness(output_tt, output_cpu)

        # Performance comparison
        speedup = avg_time_cpu / avg_time_tt
        print(f"\nPerformance Summary:")
        print(f"  TT:  {avg_time_tt:.2f} ± {std_time_tt:.2f} ms")
        print(f"  CPU: {avg_time_cpu:.2f} ± {std_time_cpu:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x {'(TT faster)' if speedup > 1 else '(CPU faster)'}")

    # Print IR dump locations
    if args.dump_ir:
        ir_path = args.ir_path or "/tmp/mlp_ir_dump"
        print("\n" + "─" * 70)
        print("IR Dumps:")
        print("─" * 70)
        print(f"Location: {ir_path}")
        print("\nTo inspect IRs:")
        print(f"  StableHLO: cat {ir_path}/stablehlo.mlir")
        print(f"  TTIR (initial): cat {ir_path}/ttir_initial.mlir")
        print(f"  TTIR (optimized): cat {ir_path}/ttir_optimized.mlir")
        print(f"  TTNN: cat {ir_path}/ttnn.mlir")
        print(f"  Flatbuffer: ls -lh {ir_path}/program.ttnn.fb")

    # Print profile location
    if args.profile:
        profile_path = args.profile_path or "/tmp/mlp_profile.json"
        print("\n" + "─" * 70)
        print("Performance Profile:")
        print("─" * 70)
        print(f"Location: {profile_path}")
        print("\nTo analyze profile:")
        print(f"  cat {profile_path} | jq '.operations[] | {{name, duration_us}}'")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
