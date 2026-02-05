"""Allow running as python -m jig."""

from jig.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
