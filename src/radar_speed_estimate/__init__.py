from .estimator import (
    EstimationError,
    VelocityEstimate,
    estimate_velocity_fused,
    estimate_velocity_from_positions,
    estimate_velocity_from_radial,
    estimate_velocity_from_time_xyvr,
    estimate_velocity_from_xyvr,
    line_of_sight_unit_vectors,
    radial_consistency_error,
)

__all__ = [
    "EstimationError",
    "VelocityEstimate",
    "estimate_velocity_fused",
    "estimate_velocity_from_positions",
    "estimate_velocity_from_radial",
    "estimate_velocity_from_time_xyvr",
    "estimate_velocity_from_xyvr",
    "line_of_sight_unit_vectors",
    "radial_consistency_error",
]


def main() -> None:
    print("radar_speed_estimate: use tests or scripts/visualize_estimation.py")
