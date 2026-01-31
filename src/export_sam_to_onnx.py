"""
ONNX Export Script for Fine-tuned SAM Model

This script exports a fine-tuned SAM (Segment Anything Model) from PyTorch (.pth)
to ONNX format (.onnx) for faster inference with ONNX Runtime.

Usage:
    python export_sam_to_onnx.py checkpoint.pth output.onnx

Requirements:
    - torch
    - transformers
    - numpy
    - onnx (for validation)
"""

import argparse
import os
from typing import Optional

# Force PyTorch-only mode to avoid TensorFlow dependency conflicts
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

import numpy as np
import torch
from transformers import SamModel, SamProcessor, SamConfig


# ================== ONNX WRAPPER CLASS ==================


class SamONNXWrapper(torch.nn.Module):
    """
    Wrapper around SamModel for ONNX export.

    Simplifies the model interface to accept only the necessary inputs
    (pixel_values and input_points) and return only pred_masks.
    """

    def __init__(self, sam_model: SamModel):
        """
        Args:
            sam_model: Fine-tuned SamModel instance
        """
        super().__init__()
        self.sam_model = sam_model

    def forward(self, pixel_values: torch.Tensor, input_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.

        Args:
            pixel_values: [B, 3, 1024, 1024] - Preprocessed RGB images
            input_points: [B, 1, N, 2] - Point prompts (N=number of points)

        Returns:
            pred_masks: [B, 1, 256, 256] - Predicted segmentation masks (logits)
        """
        outputs = self.sam_model(
            pixel_values=pixel_values,
            input_points=input_points,
            multimask_output=False
        )
        return outputs.pred_masks


# ================== EXPORT FUNCTION ==================


def export_sam_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_name: str = "facebook/sam-vit-base",
    n_points: int = 3,
    opset_version: int = 14,
    verbose: bool = True
) -> None:
    """
    Export a fine-tuned SAM model to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Path to save .onnx model (e.g., 'model.onnx')
        config_name: HuggingFace model config name (default: facebook/sam-vit-base)
        n_points: Number of point prompts (default: 3, matches training)
        opset_version: ONNX opset version (default: 14)
        verbose: Print progress messages

    Output Files:
        - {output_path}: ONNX model file
        - {output_path.replace('.onnx', '_processor_stats.npz')}: Preprocessing stats

    Example:
        >>> export_sam_to_onnx(
        ...     'models/sam_checkpoint.pth',
        ...     'models/sam_finetuned.onnx',
        ...     n_points=3
        ... )
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"Loading SAM model from {config_name}...")

    # Load model configuration
    config = SamConfig.from_pretrained(config_name)
    model = SamModel(config=config)

    if verbose:
        print(f"Loading checkpoint weights from {checkpoint_path}...")

    # Load fine-tuned weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    if verbose:
        print("Wrapping model for ONNX export...")

    # Wrap model
    wrapped_model = SamONNXWrapper(model)

    # Create dummy inputs
    batch_size = 1
    dummy_pixel_values = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32)
    dummy_input_points = torch.randn(batch_size, 1, n_points, 2, dtype=torch.float32)

    if verbose:
        print(f"Exporting to ONNX (opset {opset_version})...")
        print(f"  Input shapes:")
        print(f"    pixel_values: {list(dummy_pixel_values.shape)}")
        print(f"    input_points: {list(dummy_input_points.shape)}")

    # Export to ONNX (use legacy TorchScript exporter for SAM compatibility)
    torch.onnx.export(
        wrapped_model,
        (dummy_pixel_values, dummy_input_points),
        output_path,
        input_names=['pixel_values', 'input_points'],
        output_names=['pred_masks'],
        dynamic_axes={
            'pixel_values': {0: 'batch'},
            'input_points': {0: 'batch'},
            'pred_masks': {0: 'batch'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False  # Force legacy exporter (dynamo fails on SAM attention reshapes)
    )

    if verbose:
        print(f"✓ ONNX model saved to: {output_path}")

    # Save processor stats for inference
    processor = SamProcessor.from_pretrained(config_name)
    processor_stats = {
        'mean': np.array(processor.image_processor.image_mean, dtype=np.float32),
        'std': np.array(processor.image_processor.image_std, dtype=np.float32),
        'size': np.array(1024, dtype=np.int32)  # SAM's expected input size
    }

    stats_path = output_path.replace('.onnx', '_processor_stats.npz')
    np.savez(stats_path, **processor_stats)

    if verbose:
        print(f"✓ Processor stats saved to: {stats_path}")
        print(f"  Mean: {processor_stats['mean']}")
        print(f"  Std: {processor_stats['std']}")
        print(f"  Target size: {processor_stats['size']}")

    # Validate ONNX model
    if verbose:
        print("\nValidating ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")

            # Print model info
            print(f"\nModel Info:")
            print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
            print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")

            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.1f} MB")

        except ImportError:
            print("⚠ onnx package not installed, skipping validation")
        except Exception as e:
            print(f"⚠ Validation warning: {e}")

    print("\n✓ Export complete!")


# ================== CLI INTERFACE ==================


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned SAM model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_sam_to_onnx.py checkpoint.pth output.onnx
  python export_sam_to_onnx.py checkpoint.pth output.onnx --n_points 5
  python export_sam_to_onnx.py checkpoint.pth output.onnx --config facebook/sam-vit-huge
        """
    )

    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to .pth checkpoint file'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Path to save .onnx model file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='facebook/sam-vit-base',
        help='HuggingFace model config (default: facebook/sam-vit-base)'
    )

    parser.add_argument(
        '--n_points',
        type=int,
        default=3,
        help='Number of point prompts (default: 3)'
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Export model
    export_sam_to_onnx(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        config_name=args.config,
        n_points=args.n_points,
        opset_version=args.opset,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
