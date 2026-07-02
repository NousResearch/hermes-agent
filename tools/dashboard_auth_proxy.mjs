#!/usr/bin/env node
import http from 'node:http';
import { Buffer } from 'node:buffer';
import crypto from 'node:crypto';

const listenHost = process.env.HERMES_PROXY_HOST || '127.0.0.1';
const listenPort = Number(process.env.HERMES_PROXY_PORT || '9120');
const targetHost = process.env.HERMES_TARGET_HOST || '127.0.0.1';
const targetPort = Number(process.env.HERMES_TARGET_PORT || '9119');
const user = process.env.HERMES_PROXY_USER || 'claudio';
const password = process.env.HERMES_PROXY_PASSWORD;

if (!password) {
  console.error('Set HERMES_PROXY_PASSWORD');
  process.exit(1);
}

const expectedAuth = 'Basic ' + Buffer.from(`${user}:${password}`).toString('base64');
const sessionCookieName = 'hermes_proxy_session';
const sessionCookieValue = crypto.randomBytes(32).toString('base64url');
const sessionCookie = `${sessionCookieName}=${sessionCookieValue}`;

function hasSessionCookie(req) {
  const cookie = req.headers.cookie || '';
  return cookie.split(';').map((part) => part.trim()).includes(sessionCookie);
}

function hasBasicAuth(req) {
  return req.headers.authorization === expectedAuth;
}

function authorized(req) {
  return hasBasicAuth(req) || hasSessionCookie(req);
}

function authReason(req) {
  if (hasBasicAuth(req)) return 'basic';
  if (hasSessionCookie(req)) return 'cookie';
  return 'missing';
}

function reject(res) {
  res.writeHead(401, {
    'WWW-Authenticate': 'Basic realm="Hermes Dashboard"',
    'Content-Type': 'text/plain; charset=utf-8',
  });
  res.end('Authentication required\n');
}

function proxiedHeaders(req) {
  const headers = { ...req.headers };
  if (headers.authorization === expectedAuth) {
    delete headers.authorization;
  }
  delete headers['proxy-authorization'];
  headers.host = `${targetHost}:${targetPort}`;
  headers['x-forwarded-proto'] = 'https';
  headers['x-forwarded-host'] = req.headers.host || headers.host;
  return headers;
}

const server = http.createServer((req, res) => {
  if (!authorized(req)) {
    reject(res);
    return;
  }

  const freshLogin = hasBasicAuth(req);
  const upstream = http.request({
    host: targetHost,
    port: targetPort,
    method: req.method,
    path: req.url,
    headers: proxiedHeaders(req),
  }, (upstreamRes) => {
    const headers = { ...upstreamRes.headers };
    if (freshLogin) {
      const cookieValue = `${sessionCookie}; HttpOnly; Secure; SameSite=Lax; Path=/`;
      const existing = headers['set-cookie'];
      headers['set-cookie'] = existing ? [...(Array.isArray(existing) ? existing : [existing]), cookieValue] : [cookieValue];
    }
    res.writeHead(upstreamRes.statusCode || 502, upstreamRes.statusMessage, headers);
    upstreamRes.pipe(res);
  });

  upstream.on('error', (error) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
    }
    res.end(`Proxy error: ${error.message}\n`);
  });

  req.pipe(upstream);
});

server.on('upgrade', (req, socket, head) => {
  socket.on('error', (error) => {
    console.log(`WS client socket error ${req.url}: ${error.code || error.message}`);
  });

  if (!authorized(req)) {
    console.log(`WS ${req.url} -> 401 (${authReason(req)})`);
    socket.write('HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Basic realm="Hermes Dashboard"\r\nConnection: close\r\n\r\n');
    socket.destroy();
    return;
  }

  console.log(`WS ${req.url} -> proxy (${authReason(req)})`);

  const upstream = http.request({
    host: targetHost,
    port: targetPort,
    method: req.method,
    path: req.url,
    headers: proxiedHeaders(req),
  });

  upstream.on('upgrade', (upstreamRes, upstreamSocket, upstreamHead) => {
    console.log(`WS ${req.url} <- ${upstreamRes.statusCode || 101}`);
    upstreamSocket.on('error', (error) => {
      console.log(`WS upstream socket error ${req.url}: ${error.code || error.message}`);
      socket.destroy();
    });
    socket.write('HTTP/1.1 101 Switching Protocols\r\n');
    for (const [key, value] of Object.entries(upstreamRes.headers)) {
      if (Array.isArray(value)) {
        for (const item of value) socket.write(`${key}: ${item}\r\n`);
      } else if (value !== undefined) {
        socket.write(`${key}: ${value}\r\n`);
      }
    }
    socket.write('\r\n');
    if (upstreamHead?.length) socket.write(upstreamHead);
    if (head?.length) upstreamSocket.write(head);
    upstreamSocket.pipe(socket);
    socket.pipe(upstreamSocket);
  });

  upstream.on('response', (upstreamRes) => {
    console.log(`WS ${req.url} <- non-upgrade ${upstreamRes.statusCode}`);
    socket.write(`HTTP/1.1 ${upstreamRes.statusCode || 502} ${upstreamRes.statusMessage || 'Bad Gateway'}\r\n`);
    for (const [key, value] of Object.entries(upstreamRes.headers)) {
      if (Array.isArray(value)) {
        for (const item of value) socket.write(`${key}: ${item}\r\n`);
      } else if (value !== undefined) {
        socket.write(`${key}: ${value}\r\n`);
      }
    }
    socket.write('\r\n');
    upstreamRes.on('data', (chunk) => socket.write(chunk));
    upstreamRes.on('end', () => socket.end());
  });

  upstream.on('error', () => {
    console.log(`WS upstream request error ${req.url}`);
    socket.write('HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n');
    socket.destroy();
  });

  upstream.end();
});

server.listen(listenPort, listenHost, () => {
  console.log(`Dashboard auth proxy listening on http://${listenHost}:${listenPort} -> http://${targetHost}:${targetPort}`);
});
