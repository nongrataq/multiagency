import os
import sys
from pathlib import Path
from prometheus_client import start_http_server
from .telemetry import configure


def main():
    configure(os.getenv('DATA_DIR', 'data'))
    start_http_server(int(os.getenv('METRICS_PORT', '9100')), addr=os.getenv('METRICS_BIND', '127.0.0.1'))
    from streamlit.web import cli
    app = Path(__file__).resolve().parent.parent / 'app.py'
    sys.argv = ['streamlit', 'run', str(app), *sys.argv[1:]]
    raise SystemExit(cli.main())


if __name__ == '__main__':
    main()
