"""Run the same CLI from an installed package or a source checkout."""
from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())
