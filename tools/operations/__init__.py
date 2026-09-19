"""One live operation per owner: the table the tool thread opens into and the RPC layer drives.

An operation is anything a backend thread starts, a client answers, and the thread waits on:
a connection card, a clarify or approval prompt, an OAuth flow. Each kind owns its state and
its wire frames and implements ``Operation``; this package owns the table, first-settle-wins,
the deadline wait, cancellation by owner, and the resume snapshot.
"""

from tools.operations.protocol import Operation, Owner
from tools.operations.table import OperationAlreadyOpen, Operations, operations

__all__ = ["Operation", "OperationAlreadyOpen", "Operations", "Owner", "operations"]
