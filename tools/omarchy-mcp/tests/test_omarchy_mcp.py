import json, sys, time, os
from urllib.request import Request, urlopen
from urllib.error import HTTPError

SERVER = os.environ.get("OMARCHY_TEST_URL", "http://192.168.212.127:8911/mcp")
TOKEN = os.environ.get("OMARCHY_TEST_TOKEN", "dev-test-token-please-rotate-in-production")
TIMEOUT = 30

def init(token=TOKEN):
    body = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
    h = {"Content-Type":"application/json","Accept":"*/*"}
    if token: h["Authorization"]="Bearer "+token
    r = urlopen(Request(SERVER, data=json.dumps(body).encode(), headers=h), timeout=TIMEOUT)
    sid = r.headers.get("mcp-session-id","")
    for line in r.read().decode().split("\n"):
        if line.startswith("data:"):
            d = json.loads(line[5:])
            return sid, d["result"]
    raise RuntimeError("no data frame")

def call(method, params, sid="", token=TOKEN):
    body = {"jsonrpc":"2.0","id":2,"method":method,"params":params}
    h = {"Content-Type":"application/json","Accept":"*/*"}
    if token: h["Authorization"]="Bearer "+token
    if sid: h["mcp-session-id"] = sid
    try:
        r = urlopen(Request(SERVER, data=json.dumps(body).encode(), headers=h), timeout=TIMEOUT)
        for line in r.read().decode().split("\n"):
            if line.startswith("data:"):
                return True, json.loads(line[5:]).get("result",{})
        return True, {}
    except HTTPError as e:
        return False, {"code": e.code, "body": e.read().decode()[:200]}

tests = []
def t(fn):
    tests.append(fn)
    return fn

@t
def auth_no_token():
    ok, data = call("tools/list", {}, token=None)
    code = data.get("code",0) if isinstance(data,dict) else 0
    return code==401, "got %s" % data

@t
def auth_wrong_token():
    ok, data = call("tools/list", {}, token="wrong")
    code = data.get("code",0) if isinstance(data,dict) else 0
    return code==401, "got %s" % data

@t
def init_session():
    sid, cap = init()
    return len(sid)>0, "session=%s" % sid

@t
def tools_9():
    sid,_=init()
    ok,data=call("tools/list",{},sid=sid)
    tools=data.get("tools",[])
    return len(tools)==9, "%d tools: %s" % (len(tools),[t["name"] for t in tools])

@t
def status_ok():
    sid,_=init()
    ok,data=call("tools/call",{"name":"status","arguments":{}},sid=sid)
    text=""
    for c in (data.get("content",[]) if isinstance(data,dict) else []):
        if isinstance(c,dict) and c.get("type")=="text": text=c.get("text","")
    parsed=json.loads(text) if text else {}
    checks=all(k in parsed for k in ["ok","hostname","claude_available"])
    return ok and checks, "ok=%s host=%s" % (parsed.get("ok"),parsed.get("hostname"))

@t
def file_ops():
    sid,_=init()
    p="/tmp/om_test_%d.txt"%int(time.time())
    c="test data"
    call("tools/call",{"name":"file_write","arguments":{"path":p,"content":c}},sid=sid)
    _,d2=call("tools/call",{"name":"file_read","arguments":{"path":p}},sid=sid)
    rt="test data" in str(d2)
    _,d3=call("tools/call",{"name":"file_list","arguments":{"path":"/tmp"}},sid=sid)
    lst="om_test" in str(d3)
    _,d4=call("tools/call",{"name":"file_read","arguments":{"path":"../../etc/passwd"}},sid=sid)
    _,d5=call("tools/call",{"name":"file_read","arguments":{"path":"/etc/passwd"}},sid=sid)
    tr= "error" in str(d4).lower()
    rb= "error" in str(d5).lower()
    return rt and lst and tr and rb, "roundtrip=%s list=%s trav_block=%s root_block=%s" % (rt,lst,tr,rb)

@t
def system_run():
    sid,_=init()
    _,d1=call("tools/call",{"name":"system_run","arguments":{"command":"whoami"}},sid=sid)
    wm="jpeetz" in str(d1).lower()
    _,d2=call("tools/call",{"name":"system_run","arguments":{"command":"rm -rf /"}},sid=sid)
    bw=not d2 or "whitelist" in str(d2).lower() or "not in" in str(d2).lower()
    _,d3=call("tools/call",{"name":"system_run","arguments":{"command":"echo x; rm -rf /"}},sid=sid)
    bm=not d3 or "metachar" in str(d3).lower() or "not allowed" in str(d3).lower()
    return wm and bw and bm, "whoami_ok=%s wl_block=%s meta_block=%s" % (wm,bw,bm)

@t
def claude_hello():
    sid,_=init()
    ok,data=call("tools/call",{"name":"claude_execute","arguments":{"prompt":"say just hello and nothing else","timeout":60}},sid=sid)
    text=""
    for c in (data.get("content",[]) if isinstance(data,dict) else []):
        if isinstance(c,dict) and c.get("type")=="text": text=c.get("text","")
    return ok and "hello" in str(text).lower(), "output=%s" % text[:200]

@t
def error_handling():
    sid,_=init()
    _,d1=call("tools/call",{"name":"idonotexist","arguments":{}},sid=sid)
    _,d2=call("tools/call",{"name":"file_read","arguments":{}},sid=sid)
    unk= "error" in str(d1).lower() or "not found" in str(d1).lower()
    mis= "error" in str(d2).lower() or "required" in str(d2).lower()
    return unk and mis, "unknown_block=%s missing_block=%s" % (unk, mis)

def main():
    print("="*60)
    print("Omarchy MCP Server test suite -> %s" % SERVER)
    print("="*60)
    passed=0
    failed=[]
    for fn in tests:
        try:
            ok,detail=fn()
        except Exception as e:
            ok,detail=False,"EXCEPTION: %s"%e
        status="PASS" if ok else "FAIL"
        print("[%s] %s" % (status, fn.__name__))
        print("       %s" % detail)
        if ok: passed+=1
        else: failed.append(fn.__name__)
    print("="*60)
    print("Results: %d/%d passed" % (passed, len(tests)))
    if failed: print("Failed: %s" % ", ".join(failed))
    print("="*60)
    sys.exit(0 if passed==len(tests) else 1)

if __name__=="__main__":
    main()
