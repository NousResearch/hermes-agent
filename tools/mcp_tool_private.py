"""Single-use private attachment on an exact outgoing MCP HTTP request.

The transport owns a pending request slot under its existing RPC lock. Headers
are set on only the matching HTTP request, never the shared client's defaults.
"""
import json
from contextlib import contextmanager

from hermes_cli.private_invocation_store import DESCRIPTOR, digest


class PrivateCall:
    __slots__ = ('issuer', 'reference', 'server', 'operation', 'action', 'used', '_wire_ref')

    def __init__(self, issuer, reference, server, operation, action):
        self.issuer, self.reference, self.server = issuer, reference, server
        self.operation, self.action, self.used = operation, action, False
        self._wire_ref = None

    def __repr__(self):
        return '<private MCP request>'

    def __reduce__(self):
        raise TypeError('private capability is not serializable')

    @contextmanager
    def attach(self, server, tool, args):
        if self.used:
            raise PermissionError('private operation already attempted; reconcile unknown effects')
        self.used = True
        from importlib.metadata import version
        if (version('mcp'), version('httpx2')) != ('2.3.0', '2.15.1'):
            raise PermissionError('private carrier requires pinned mcp2.3.0/httpx2 2.15.1')
        caps = getattr(server.initialize_result, 'capabilities', None)
        experimental = getattr(caps, 'experimental', None) or {}
        descriptor = experimental.get('privateInvocation')
        if (type(descriptor) is not dict or type(descriptor.get('version')) is not int
                or descriptor != DESCRIPTOR
                or not getattr(server, '_private_carrier_ready', False)):
            raise PermissionError('host lacks privateInvocation v1 request carrier')
        if (tool != 'call' or type(args) is not dict or set(args) != {'calls'}
                or type(args['calls']) is not dict or set(args['calls']) != {'path', 'input'}):
            raise PermissionError('private v1 requires one exact direct call')
        target = self.issuer.targets[self.server]
        if args['calls']['path'] != target['toolPath']:
            raise PermissionError('configured tool path mismatch')
        ref = self.issuer.issue(self.reference, self.server, self.operation, self.action, args['calls']['input'])
        self._wire_ref = ref
        ticket = _Ticket(ref, digest({'name': tool, 'arguments': args}))
        if getattr(server, '_private_request', None) is not None:
            raise PermissionError('overlapping private request')
        server._private_request = ticket
        try:
            yield
            if not ticket.sent:
                raise PermissionError('private carrier did not observe exact request')
        finally:
            server._private_request = None


class _Ticket:
    __slots__ = ('reference', 'digest', 'sent')

    def __init__(self, reference, expected):
        self.reference, self.digest, self.sent = reference, expected, False


def request_hook(server, endpoint):
    """Installed only on the owned modern streamable HTTP client."""
    async def attach(request):
        # Never accept configured static references, including on ordinary calls.
        if 'Private-Invocation' in request.headers:
            raise PermissionError('static private invocation headers forbidden')
        ticket = getattr(server, '_private_request', None)
        if ticket is None:
            return
        if str(request.url) != endpoint or request.method != 'POST':
            raise PermissionError('private request cannot redirect or change endpoint')
        body = json.loads(request.content)
        if body.get('method') != 'tools/call':
            return  # handshake/progress cannot inherit a permit
        params = body.get('params', {})
        # Progress metadata is not action input. No other metadata carries authority.
        actual = {'name': params.get('name'), 'arguments': params.get('arguments')}
        if ticket.sent or digest(actual) != ticket.digest:
            raise PermissionError('private final request mismatch or replay')
        ticket.sent = True
        request.headers['Private-Invocation'] = ticket.reference
    return attach


def response_hook(server):
    async def check(response):
        if getattr(server, '_private_request', None) is not None and 300 <= response.status_code < 400:
            raise PermissionError('private invocation redirects forbidden')
    return check
