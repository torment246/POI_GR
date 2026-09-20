"""Allow ``python -m qg_prqk`` to invoke the public CLI."""

from qg_prqk.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
