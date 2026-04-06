"""
Prediction mode enumeration for DataMint models.
"""

from enum import Enum


class PredictionMode(str, Enum):
    """
    Enumeration of supported prediction modes.

    Each mode corresponds to a specific method signature in DatamintModel.
    """
    # Standard modes
    DEFAULT = 'default'                # Default: process entire resource as-is

    # Simple modes
    IMAGE = 'image'                    # Process single 2d image resource

    # Video/temporal modes
    FRAME = 'frame'                    # Extract and process specific frame
    FRAME_RANGE = 'frame_range'        # Process contiguous frame range
    ALL_FRAMES = 'all_frames'          # Process all frames independently
    TEMPORAL_SEQUENCE = 'temporal_sequence'  # Process with temporal context window

    # 3D volume modes
    SLICE = 'slice'                    # Extract and process specific slice
    SLICE_RANGE = 'slice_range'        # Process contiguous slice range
    PRIMARY_SLICE = 'primary_slice'    # Process center/primary slice
    # MULTI_PLANE = 'multi_plane'        # Process multiple anatomical planes
    VOLUME = 'volume'                  # Process entire 3D volume

    # Spatial modes
    # ROI = 'roi'                        # Process single region of interest
    # MULTI_ROI = 'multi_roi'            # Process multiple regions
    # TILE = 'tile'                      # Split into tiles (whole slide imaging)
    # PATCH = 'patch'                    # Extract patches around points

    # Advanced modes
    INTERACTIVE = 'interactive'        # With user prompts (SAM-like)
    FEW_SHOT = 'few_shot'             # With context examples
    # MULTI_VIEW = 'multi_view'          # Multiple views of same subject
