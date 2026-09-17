"""Interactive Brokers connectivity.

All IB access is confined to this module. Notebooks and other package code never
connect to IB directly — they call a loader here, which reads the parquet cache
under `data/` and only touches the API when `refresh=True` is passed.

Connection settings (host, port, client id) come from a `.env` file (see
`.env.example`), never from code, per the licence terms and to keep the paper vs.
live account switch out of source control.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ib_async import IB

load_dotenv()


def get_connection_settings() -> tuple[str, int, int]:
    """Read (host, port, client_id) from the environment.

    Raises `KeyError` if `.env` hasn't been set up — see `.env.example`.
    """
    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ["IB_PORT"])
    client_id = int(os.environ.get("IB_CLIENT_ID", "1"))
    return host, port, client_id


def connect(timeout: float = 10.0) -> IB:
    """Connect to TWS / IB Gateway using the settings in `.env`.

    Caller is responsible for calling `ib.disconnect()` when done (or use it as
    a context manager via `ib_async`'s `IB.connect` / `with connect() as ib:`
    is not supported directly by `ib_async`, so prefer a try/finally).
    """
    host, port, client_id = get_connection_settings()
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=timeout)
    return ib


def ping() -> dict[str, str]:
    """Connect, ask the server for the current time, and disconnect.

    A minimal round trip that proves TWS / IB Gateway is up, logged in, and
    reachable on the configured port — without touching market data
    subscriptions or pacing limits.
    """
    ib = connect()
    try:
        server_time = ib.reqCurrentTime()
        accounts = ib.managedAccounts()
        return {
            "server_time": str(server_time),
            "accounts": ", ".join(accounts),
            "server_version": str(ib.client.serverVersion()),
        }
    finally:
        ib.disconnect()


if __name__ == "__main__":
    result = ping()
    print("Connected to IB.")
    for key, value in result.items():
        print(f"  {key}: {value}")
