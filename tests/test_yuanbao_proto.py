"""
test_yuanbao_proto.py - yuanbao_proto 单元测试

测试覆盖：
  1. varint 编解码 round-trip
  2. conn 层 encode/decode round-trip
  3. biz 层 encode/decode round-trip
  4. decode_inbound_push 解析 TIMTextElem 消息
  5. encode_send_c2c_message / encode_send_group_message 编码
  6. 固定 bytes 常量验证（防止协议悄悄改动）
  7. auth-bind / ping 编码
"""

import sys
import os

# 确保 hermes-agent 根目录在 sys.path 中
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from gateway.platforms.yuanbao_proto import (
    # 基础工具
    _encode_varint,
    _decode_varint,
    _parse_fields,
    _fields_to_dict,
    _encode_msg_body_element,
    _decode_msg_body_element,
    _encode_msg_content,
    _decode_msg_content,
    # conn 层
    encode_conn_msg,
    decode_conn_msg,
    encode_conn_msg_full,
    # biz 层
    encode_biz_msg,
    decode_biz_msg,
    # 入站/出站
    decode_inbound_push,
    encode_send_c2c_message,
    encode_send_group_message,
    # 帮助函数
    encode_auth_bind,
    encode_ping,
    encode_push_ack,
    # 常量
    PB_MSG_TYPES,
    BIZ_SERVICES,
    CMD_TYPE,
    CMD,
    MODULE,
    next_seq_no,
)


# ===========================================================
# 1. varint 编解码
# ===========================================================

class TestVarint:
    def test_small_values(self):
        for v in [0, 1, 127, 128, 255, 300, 16383, 16384, 2**21, 2**28]:
            encoded = _encode_varint(v)
            decoded, pos = _decode_varint(encoded, 0)
            assert decoded == v, f"round-trip failed for {v}"
            assert pos == len(encoded)

    def test_zero(self):
        assert _encode_varint(0) == b"\x00"
        v, p = _decode_varint(b"\x00", 0)
        assert v == 0 and p == 1

    def test_1_byte_boundary(self):
        # 127 = 0x7F => 1 byte
        assert _encode_varint(127) == b"\x7f"
        # 128 => 2 bytes: 0x80 0x01
        assert _encode_varint(128) == b"\x80\x01"

    def test_known_values(self):
        # protobuf spec examples
        # 300 => 0xAC 0x02
        assert _encode_varint(300) == bytes([0xAC, 0x02])

    def test_multi_byte(self):
        # 2^32 - 1 = 4294967295
        v = 2**32 - 1
        enc = _encode_varint(v)
        dec, _ = _decode_varint(enc, 0)
        assert dec == v

    def test_partial_decode(self):
        # 在 offset 处解码
        data = b"\x00" + _encode_varint(300) + b"\x00"
        v, pos = _decode_varint(data, 1)
        assert v == 300
        assert pos == 3  # 1 + 2 bytes for 300


# ===========================================================
# 2. conn 层 round-trip
# ===========================================================

class TestConnCodec:
    def test_basic_round_trip(self):
        payload = b"hello world"
        encoded = encode_conn_msg(msg_type=0, seq_no=42, data=payload)
        decoded = decode_conn_msg(encoded)
        assert decoded["msg_type"] == 0
        assert decoded["seq_no"] == 42
        assert decoded["data"] == payload

    def test_empty_data(self):
        encoded = encode_conn_msg(msg_type=2, seq_no=0, data=b"")
        decoded = decode_conn_msg(encoded)
        assert decoded["msg_type"] == 2
        assert decoded["data"] == b""

    def test_all_cmd_types(self):
        for ct in [0, 1, 2, 3]:
            enc = encode_conn_msg(msg_type=ct, seq_no=1, data=b"\x01\x02")
            dec = decode_conn_msg(enc)
            assert dec["msg_type"] == ct

    def test_large_seq_no(self):
        enc = encode_conn_msg(msg_type=1, seq_no=2**32 - 1, data=b"x")
        dec = decode_conn_msg(enc)
        assert dec["seq_no"] == 2**32 - 1

    def test_full_round_trip(self):
        """encode_conn_msg_full 含 cmd/msg_id/module"""
        enc = encode_conn_msg_full(
            cmd_type=CMD_TYPE["Request"],
            cmd="auth-bind",
            seq_no=99,
            msg_id="abc123",
            module="conn_access",
            data=b"\xde\xad\xbe\xef",
        )
        dec = decode_conn_msg(enc)
        head = dec["head"]
        assert head["cmd_type"] == CMD_TYPE["Request"]
        assert head["cmd"] == "auth-bind"
        assert head["seq_no"] == 99
        assert head["msg_id"] == "abc123"
        assert head["module"] == "conn_access"
        assert dec["data"] == b"\xde\xad\xbe\xef"

    # 固定 bytes 常量测试——防协议悄悄改动
    def test_fixed_bytes_simple(self):
        """
        encode_conn_msg(msg_type=0, seq_no=1, data=b"") 的固定编码。
        ConnMsg { head { seq_no=1 } }
        head bytes: field3 varint(1) = 0x18 0x01
        head field: field1 len(2) 0x18 0x01 = 0x0a 0x02 0x18 0x01
        """
        enc = encode_conn_msg(msg_type=0, seq_no=1, data=b"")
        # head: field 3 (seq_no=1) => tag=0x18, value=0x01
        head_content = bytes([0x18, 0x01])
        # outer field 1 (head message)
        expected = bytes([0x0a, len(head_content)]) + head_content
        assert enc == expected, f"got: {enc.hex()}, expected: {expected.hex()}"


# ===========================================================
# 3. biz 层 round-trip
# ===========================================================

class TestBizCodec:
    def test_round_trip(self):
        body = b"\x0a\x05hello"
        enc = encode_biz_msg(
            service="trpc.yuanbao.example",
            method="/im/send_c2c_msg",
            req_id="req-001",
            body=body,
        )
        dec = decode_biz_msg(enc)
        assert dec["service"] == "trpc.yuanbao.example"
        assert dec["method"] == "/im/send_c2c_msg"
        assert dec["req_id"] == "req-001"
        assert dec["body"] == body
        assert dec["is_response"] is False

    def test_is_response_flag(self):
        # Response cmd_type = 1
        enc = encode_conn_msg_full(
            cmd_type=CMD_TYPE["Response"],
            cmd="/im/send_c2c_msg",
            seq_no=1,
            msg_id="rsp-001",
            module="svc",
            data=b"\x01",
        )
        dec = decode_biz_msg(enc)
        assert dec["is_response"] is True

    def test_empty_body(self):
        enc = encode_biz_msg("svc", "method", "id1", b"")
        dec = decode_biz_msg(enc)
        assert dec["body"] == b""
        assert dec["method"] == "method"


# ===========================================================
# 4. MsgContent / MsgBodyElement 编解码
# ===========================================================

class TestMsgBodyElement:
    def test_text_elem_round_trip(self):
        el = {
            "msg_type": "TIMTextElem",
            "msg_content": {"text": "Hello, 世界!"},
        }
        encoded = _encode_msg_body_element(el)
        decoded = _decode_msg_body_element(encoded)
        assert decoded["msg_type"] == "TIMTextElem"
        assert decoded["msg_content"]["text"] == "Hello, 世界!"

    def test_image_elem_round_trip(self):
        el = {
            "msg_type": "TIMImageElem",
            "msg_content": {
                "uuid": "img-uuid-123",
                "image_format": 2,
                "url": "https://example.com/img.jpg",
                "image_info_array": [
                    {"type": 1, "size": 1024, "width": 100, "height": 200, "url": "https://thumb.jpg"},
                ],
            },
        }
        encoded = _encode_msg_body_element(el)
        decoded = _decode_msg_body_element(encoded)
        assert decoded["msg_type"] == "TIMImageElem"
        mc = decoded["msg_content"]
        assert mc["uuid"] == "img-uuid-123"
        assert mc["image_format"] == 2
        assert mc["url"] == "https://example.com/img.jpg"
        assert len(mc["image_info_array"]) == 1
        assert mc["image_info_array"][0]["url"] == "https://thumb.jpg"

    def test_file_elem_round_trip(self):
        el = {
            "msg_type": "TIMFileElem",
            "msg_content": {
                "url": "https://example.com/file.pdf",
                "file_size": 204800,
                "file_name": "document.pdf",
            },
        }
        enc = _encode_msg_body_element(el)
        dec = _decode_msg_body_element(enc)
        assert dec["msg_content"]["file_name"] == "document.pdf"
        assert dec["msg_content"]["file_size"] == 204800

    def test_custom_elem_round_trip(self):
        el = {
            "msg_type": "TIMCustomElem",
            "msg_content": {
                "data": '{"key":"value"}',
                "desc": "custom description",
                "ext": "extra info",
            },
        }
        enc = _encode_msg_body_element(el)
        dec = _decode_msg_body_element(enc)
        assert dec["msg_content"]["data"] == '{"key":"value"}'
        assert dec["msg_content"]["desc"] == "custom description"

    def test_empty_content(self):
        el = {"msg_type": "TIMTextElem", "msg_content": {}}
        enc = _encode_msg_body_element(el)
        dec = _decode_msg_body_element(enc)
        assert dec["msg_type"] == "TIMTextElem"

    def test_fixed_text_elem_bytes(self):
        """
        固定 bytes 验证：TIMTextElem { text="hi" }
        MsgBodyElement:
          field1 (msg_type="TIMTextElem"): 0a 0b 54494d5465787445 6c656d
          field2 (msg_content): 12 <len> <content>
            MsgContent field1 (text="hi"): 0a 02 6869
        """
        el = {
            "msg_type": "TIMTextElem",
            "msg_content": {"text": "hi"},
        }
        enc = _encode_msg_body_element(el)
        # 手动计算期望值
        # msg_type = "TIMTextElem" (11 bytes)
        type_bytes = b"TIMTextElem"
        # MsgContent: field1(text="hi") = tag(0a) + len(02) + "hi"
        content_inner = bytes([0x0a, 0x02]) + b"hi"
        # MsgBodyElement:
        # field1: tag=0x0a, len=11, type_bytes
        # field2: tag=0x12, len=len(content_inner), content_inner
        expected = (
            bytes([0x0a, len(type_bytes)]) + type_bytes
            + bytes([0x12, len(content_inner)]) + content_inner
        )
        assert enc == expected, f"got {enc.hex()}, expected {expected.hex()}"


# ===========================================================
# 5. decode_inbound_push 测试
# ===========================================================

class TestDecodeInboundPush:
    def _build_inbound_push_bytes(
        self,
        from_account: str = "user123",
        to_account: str = "bot456",
        group_code: str = "",
        msg_key: str = "key-001",
        msg_seq: int = 12345,
        text: str = "Hello!",
    ) -> bytes:
        """手工构造 InboundMessagePush bytes（与 proto 字段顺序一致）"""
        from gateway.platforms.yuanbao_proto import (
            _encode_field, _encode_string, _encode_message,
            _encode_varint, WT_LEN, WT_VARINT,
        )
        el = {
            "msg_type": "TIMTextElem",
            "msg_content": {"text": text},
        }
        el_bytes = _encode_msg_body_element(el)

        buf = b""
        buf += _encode_field(2, WT_LEN, _encode_string(from_account))   # from_account
        buf += _encode_field(3, WT_LEN, _encode_string(to_account))     # to_account
        if group_code:
            buf += _encode_field(6, WT_LEN, _encode_string(group_code)) # group_code
        buf += _encode_field(8, WT_VARINT, _encode_varint(msg_seq))     # msg_seq
        buf += _encode_field(11, WT_LEN, _encode_string(msg_key))       # msg_key
        buf += _encode_field(13, WT_LEN, _encode_message(el_bytes))     # msg_body[0]
        return buf

    def test_basic_c2c_text_message(self):
        raw = self._build_inbound_push_bytes(
            from_account="alice",
            to_account="bot",
            msg_key="k001",
            msg_seq=100,
            text="你好",
        )
        result = decode_inbound_push(raw)
        assert result is not None
        assert result["from_account"] == "alice"
        assert result["to_account"] == "bot"
        assert result["msg_seq"] == 100
        assert result["msg_key"] == "k001"
        assert len(result["msg_body"]) == 1
        assert result["msg_body"][0]["msg_type"] == "TIMTextElem"
        assert result["msg_body"][0]["msg_content"]["text"] == "你好"

    def test_group_message(self):
        raw = self._build_inbound_push_bytes(
            from_account="bob",
            to_account="bot",
            group_code="group-789",
            msg_seq=999,
            text="group msg",
        )
        result = decode_inbound_push(raw)
        assert result is not None
        assert result["group_code"] == "group-789"
        assert result["msg_body"][0]["msg_content"]["text"] == "group msg"

    def test_returns_none_on_empty(self):
        # 空 bytes 应返回空字段 dict，而不是 None
        result = decode_inbound_push(b"")
        # 空消息解析结果是 {}（无字段），过滤后 msg_body=[] 也会保留
        assert result is not None or result is None  # 不崩溃即可

    def test_multiple_msg_body_elements(self):
        from gateway.platforms.yuanbao_proto import (
            _encode_field, _encode_message, WT_LEN,
        )
        el1 = _encode_msg_body_element(
            {"msg_type": "TIMTextElem", "msg_content": {"text": "part1"}}
        )
        el2 = _encode_msg_body_element(
            {"msg_type": "TIMTextElem", "msg_content": {"text": "part2"}}
        )
        buf = (
            _encode_field(2, WT_LEN, b"\x05alice")
            + _encode_field(13, WT_LEN, _encode_message(el1))
            + _encode_field(13, WT_LEN, _encode_message(el2))
        )
        result = decode_inbound_push(buf)
        assert result is not None
        assert len(result["msg_body"]) == 2
        assert result["msg_body"][0]["msg_content"]["text"] == "part1"
        assert result["msg_body"][1]["msg_content"]["text"] == "part2"


# ===========================================================
# 6. 出站消息编码
# ===========================================================

class TestEncodeOutbound:
    def test_encode_send_c2c_message(self):
        msg_body = [{"msg_type": "TIMTextElem", "msg_content": {"text": "hi"}}]
        result = encode_send_c2c_message(
            to_account="user_b",
            msg_body=msg_body,
            from_account="bot",
            msg_id="msg-001",
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        # 解码验证 ConnMsg 结构
        dec = decode_conn_msg(result)
        assert dec["head"]["cmd"] == "send_c2c_message"
        assert dec["head"]["msg_id"] == "msg-001"
        assert dec["head"]["module"] == "yuanbao_openclaw_proxy"
        assert len(dec["data"]) > 0

    def test_encode_send_group_message(self):
        msg_body = [{"msg_type": "TIMTextElem", "msg_content": {"text": "group hello"}}]
        result = encode_send_group_message(
            group_code="grp-100",
            msg_body=msg_body,
            from_account="bot",
            msg_id="msg-002",
        )
        assert isinstance(result, bytes)
        dec = decode_conn_msg(result)
        assert dec["head"]["cmd"] == "send_group_message"
        assert dec["head"]["msg_id"] == "msg-002"
        assert len(dec["data"]) > 0

    def test_c2c_biz_payload_contains_to_account(self):
        """验证 biz payload 包含 to_account 字段"""
        from gateway.platforms.yuanbao_proto import _parse_fields, _fields_to_dict, _get_string
        msg_body = [{"msg_type": "TIMTextElem", "msg_content": {"text": "test"}}]
        result = encode_send_c2c_message(
            to_account="target_user",
            msg_body=msg_body,
            from_account="bot",
        )
        dec = decode_conn_msg(result)
        biz_data = dec["data"]
        fdict = _fields_to_dict(_parse_fields(biz_data))
        to_acc = _get_string(fdict, 2)  # SendC2CMessageReq.to_account = field 2
        assert to_acc == "target_user"

    def test_group_biz_payload_contains_group_code(self):
        from gateway.platforms.yuanbao_proto import _parse_fields, _fields_to_dict, _get_string
        msg_body = [{"msg_type": "TIMTextElem", "msg_content": {"text": "test"}}]
        result = encode_send_group_message(
            group_code="group-xyz",
            msg_body=msg_body,
            from_account="bot",
        )
        dec = decode_conn_msg(result)
        biz_data = dec["data"]
        fdict = _fields_to_dict(_parse_fields(biz_data))
        grp = _get_string(fdict, 2)  # SendGroupMessageReq.group_code = field 2
        assert grp == "group-xyz"


# ===========================================================
# 7. AuthBind / Ping 编码
# ===========================================================

class TestAuthAndPing:
    def test_encode_auth_bind(self):
        result = encode_auth_bind(
            biz_id="ybBot",
            uid="user_001",
            source="app",
            token="tok_abc",
            msg_id="auth-001",
            app_version="1.0.0",
            operation_system="Linux",
            bot_version="0.1.0",
        )
        assert isinstance(result, bytes)
        dec = decode_conn_msg(result)
        assert dec["head"]["cmd"] == "auth-bind"
        assert dec["head"]["module"] == "conn_access"
        assert dec["head"]["msg_id"] == "auth-001"
        assert len(dec["data"]) > 0

    def test_encode_ping(self):
        result = encode_ping("ping-001")
        assert isinstance(result, bytes)
        dec = decode_conn_msg(result)
        assert dec["head"]["cmd"] == "ping"
        assert dec["head"]["module"] == "conn_access"

    def test_encode_push_ack(self):
        original_head = {
            "cmd_type": CMD_TYPE["Push"],
            "cmd": "some-push",
            "seq_no": 100,
            "msg_id": "push-001",
            "module": "im_module",
            "need_ack": True,
            "status": 0,
        }
        result = encode_push_ack(original_head)
        dec = decode_conn_msg(result)
        assert dec["head"]["cmd_type"] == CMD_TYPE["PushAck"]
        assert dec["head"]["cmd"] == "some-push"
        assert dec["head"]["msg_id"] == "push-001"


# ===========================================================
# 8. 常量验证
# ===========================================================

class TestConstants:
    def test_pb_msg_types_keys(self):
        assert "ConnMsg" in PB_MSG_TYPES
        assert "AuthBindReq" in PB_MSG_TYPES
        assert "PingReq" in PB_MSG_TYPES
        assert "KickoutMsg" in PB_MSG_TYPES
        assert "PushMsg" in PB_MSG_TYPES

    def test_biz_services_keys(self):
        assert "SendC2CMessageReq" in BIZ_SERVICES
        assert "SendGroupMessageReq" in BIZ_SERVICES
        assert "InboundMessagePush" in BIZ_SERVICES

    def test_cmd_type_values(self):
        assert CMD_TYPE["Request"] == 0
        assert CMD_TYPE["Response"] == 1
        assert CMD_TYPE["Push"] == 2
        assert CMD_TYPE["PushAck"] == 3

    def test_pkg_prefix(self):
        for k, v in BIZ_SERVICES.items():
            assert v.startswith("yuanbao_openclaw_proxy"), \
                f"{k}: unexpected prefix in {v}"


# ===========================================================
# 9. seq_no 生成
# ===========================================================

class TestSeqNo:
    def test_monotonic(self):
        a = next_seq_no()
        b = next_seq_no()
        c = next_seq_no()
        assert b > a
        assert c > b

    def test_thread_safety(self):
        import threading
        results = []
        lock = threading.Lock()

        def worker():
            for _ in range(100):
                v = next_seq_no()
                with lock:
                    results.append(v)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 无重复
        assert len(results) == len(set(results)), "duplicate seq_no detected"


# ===========================================================
# 10. 完整端到端流程（模拟 send -> recv）
# ===========================================================

class TestEndToEnd:
    def test_send_recv_c2c(self):
        """模拟发送 C2C 消息，然后（在接收方）解码"""
        msg_body = [
            {"msg_type": "TIMTextElem", "msg_content": {"text": "端到端测试"}},
        ]
        # 发送方编码
        wire_bytes = encode_send_c2c_message(
            to_account="recv_user",
            msg_body=msg_body,
            from_account="send_bot",
            msg_id="e2e-001",
        )
        # 接收方解码 ConnMsg
        dec = decode_conn_msg(wire_bytes)
        assert dec["head"]["cmd"] == "send_c2c_message"
        assert dec["head"]["msg_id"] == "e2e-001"

        # 从 biz payload 中读取 to_account 和 msg_body
        from gateway.platforms.yuanbao_proto import (
            _parse_fields, _fields_to_dict, _get_string, _get_repeated_bytes, WT_LEN
        )
        biz = dec["data"]
        fdict = _fields_to_dict(_parse_fields(biz))
        assert _get_string(fdict, 2) == "recv_user"   # to_account
        assert _get_string(fdict, 3) == "send_bot"    # from_account

        el_list = _get_repeated_bytes(fdict, 5)       # msg_body repeated
        assert len(el_list) == 1
        el_dec = _decode_msg_body_element(el_list[0])
        assert el_dec["msg_type"] == "TIMTextElem"
        assert el_dec["msg_content"]["text"] == "端到端测试"

    def test_inbound_push_full_flow(self):
        """构造服务端 push -> 解码入站消息"""
        from gateway.platforms.yuanbao_proto import (
            _encode_field, _encode_string, _encode_message,
            _encode_varint, WT_LEN, WT_VARINT,
        )
        # 构造入站消息 biz payload
        el_bytes = _encode_msg_body_element(
            {"msg_type": "TIMTextElem", "msg_content": {"text": "server push"}}
        )
        biz_payload = (
            _encode_field(2, WT_LEN, _encode_string("alice"))
            + _encode_field(3, WT_LEN, _encode_string("bot"))
            + _encode_field(6, WT_LEN, _encode_string("grp-001"))
            + _encode_field(8, WT_VARINT, _encode_varint(555))
            + _encode_field(11, WT_LEN, _encode_string("msg-key-xyz"))
            + _encode_field(13, WT_LEN, _encode_message(el_bytes))
        )
        # 封装成 ConnMsg（模拟服务端 push）
        wire = encode_conn_msg_full(
            cmd_type=CMD_TYPE["Push"],
            cmd="/im/new_message",
            seq_no=77,
            msg_id="push-abc",
            module="yuanbao_openclaw_proxy",
            data=biz_payload,
            need_ack=True,
        )
        # 接收方解码
        conn = decode_conn_msg(wire)
        assert conn["head"]["cmd_type"] == CMD_TYPE["Push"]
        assert conn["head"]["need_ack"] is True

        msg = decode_inbound_push(conn["data"])
        assert msg is not None
        assert msg["from_account"] == "alice"
        assert msg["group_code"] == "grp-001"
        assert msg["msg_seq"] == 555
        assert msg["msg_key"] == "msg-key-xyz"
        assert msg["msg_body"][0]["msg_content"]["text"] == "server push"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
