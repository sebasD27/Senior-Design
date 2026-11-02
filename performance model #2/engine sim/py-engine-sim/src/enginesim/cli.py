"""Command-line interface for quick engine evaluations."""

import argparse
import json

from .presets import p1000_like
from .turbojet import FlightCondition, evaluate_turbojet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reduced-order turbojet model")
    parser.add_argument("--preset", choices=["p1000"], default="p1000")
    parser.add_argument("--altitude-ft", type=float, default=0.0)
    parser.add_argument("--mach", type=float, default=0.0)
    parser.add_argument("--throttle", type=float, default=1.0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    config = p1000_like()
    result = evaluate_turbojet(config, FlightCondition(args.altitude_ft, args.mach), args.throttle)
    indent = 2 if args.pretty else None
    print(json.dumps(result.as_dict(), indent=indent))


if __name__ == "__main__":
    main()
