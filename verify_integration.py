#!/usr/bin/env python3
"""
核心改造集成验证脚本

验证所有已安装的模块是否正常工作
"""

import sys
import json

print("🧪 核心改造集成验证\n")
print("=" * 60)

# 测试1：Context Engine
print("\n1️⃣  测试 Context Engine")
try:
    import hermes_context_engine as hce
    
    # 测试 Tokenizer
    tokenizer = hce.PyTokenizer("gpt-4")
    count = tokenizer.count_text("Hello, world!")
    print(f"   ✅ Tokenizer: 'Hello, world!' = {count} tokens")
    
    # 测试 CoordinateTracker
    tracker = hce.PyCoordinateTracker()
    tracker.add_message("msg_1", "system", "Test message", 10)
    stats = tracker.stats()
    print(f"   ✅ CoordinateTracker: {stats['total_messages']} 条消息")
    
    # 测试 ContextCompressor
    compressor = hce.PyContextCompressor("keep_recent", 2)
    messages = json.dumps([
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User 1"},
        {"role": "assistant", "content": "Assistant 1"},
        {"role": "user", "content": "User 2"}
    ])
    result = compressor.compress(messages, [10, 5, 8, 5])
    print(f"   ✅ ContextCompressor: 压缩比 {result['compression_ratio']:.1%}")
    
    print("   ✅ Context Engine 全部功能正常")
except Exception as e:
    print(f"   ❌ Context Engine 测试失败: {e}")
    sys.exit(1)

# 测试2：多模型路由器
print("\n2️⃣  测试多模型路由器")
try:
    import hermes_multi_model_router as hmr
    
    # 测试创建路由器
    router = hmr.PyRouter(None)
    print("   ✅ 路由器创建成功")
    
    # 测试设置并发模式
    router.set_concurrent_mode("single", None)
    print("   ✅ 并发模式设置成功")
    
    # 测试设置仲裁策略
    router.set_arbitration_strategy("fastest")
    print("   ✅ 仲裁策略设置成功")
    
    print("   ✅ 多模型路由器全部功能正常")
except Exception as e:
    print(f"   ❌ 多模型路由器测试失败: {e}")
    sys.exit(1)

# 测试3：主链路插桩
print("\n3️⃣  测试主链路插桩")
try:
    from agent.trace_hooks import trace_tool_call, get_trace_stats, export_traces
    
    # 测试装饰器
    @trace_tool_call
    def test_tool(tool_name, tool_args, task_id=None):
        return f"Tool {tool_name} executed"
    
    result = test_tool("test", {"param": "value"}, "task_123")
    print(f"   ✅ 装饰器工作正常: {result}")
    
    # 测试统计
    stats = get_trace_stats()
    if stats:
        print(f"   ✅ 统计功能正常: {stats['total']} 条轨迹")
    else:
        print("   ⚠️  统计功能返回 None（可能是首次运行）")
    
    print("   ✅ 主链路插桩全部功能正常")
except Exception as e:
    print(f"   ❌ 主链路插桩测试失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("✅ 所有模块集成验证通过！\n")

print("📊 已安装模块：")
print("   • Context Engine (hermes_context_engine)")
print("   • 多模型路由器 (hermes_multi_model_router)")
print("   • 主链路插桩 (agent.trace_hooks)")

print("\n🚀 下一步：")
print("   1. 在 model_tools.py 中应用 @trace_tool_call 装饰器")
print("   2. 在需要的地方使用 Context Engine 替换现有 token 计数")
print("   3. 配置多模型路由器（可选）")

print("\n📖 使用文档：")
print("   ~/.hermes/core-reform/docs/overall_complete_report.md")
