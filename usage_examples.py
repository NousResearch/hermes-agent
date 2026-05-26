#!/usr/bin/env python3
"""
核心改造功能使用示例

演示如何使用所有已安装的核心改造模块
"""

import json
import sys

print("=" * 70)
print("核心改造功能使用示例")
print("=" * 70)

# ============================================================================
# 示例1：使用 Context Engine 进行精确 token 计数
# ============================================================================
print("\n📊 示例1：精确 Token 计数")
print("-" * 70)

try:
    import hermes_context_engine as hce
    
    # 创建 tokenizer
    tokenizer = hce.PyTokenizer("gpt-4")
    
    # 测试不同长度的文本
    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能正在改变世界。",
    ]
    
    for text in texts:
        count = tokenizer.count_text(text)
        print(f"  文本: {text}")
        print(f"  Token 数: {count}")
        print()
    
    # 计算消息列表的 token 数
    messages = json.dumps([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ])
    
    result = tokenizer.count_messages(messages)
    print(f"  消息列表总 token 数: {result['total']}")
    print(f"  按角色分组:")
    for role, count in result['by_role'].items():
        print(f"    {role}: {count} tokens")
    
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ============================================================================
# 示例2：使用 CoordinateTracker 追踪消息位置
# ============================================================================
print("\n📍 示例2：消息坐标追踪")
print("-" * 70)

try:
    import hermes_context_engine as hce
    
    # 创建 tracker
    tracker = hce.PyCoordinateTracker()
    
    # 添加多条消息
    messages = [
        ("msg_1", "system", "You are a helpful assistant.", 10),
        ("msg_2", "user", "Hello!\nHow are you?", 8),
        ("msg_3", "assistant", "I'm doing well, thank you!", 7),
    ]
    
    for msg_id, role, content, tokens in messages:
        tracker.add_message(msg_id, role, content, tokens)
        print(f"  添加消息: {msg_id} ({role})")
    
    print()
    
    # 查询坐标
    coord = tracker.get_coordinate("msg_2")
    if coord:
        print(f"  消息 'msg_2' 的坐标:")
        print(f"    起始行: {coord['start_line']}")
        print(f"    结束行: {coord['end_line']}")
        print(f"    起始字符: {coord['start_char']}")
        print(f"    结束字符: {coord['end_char']}")
        print(f"    Token 数: {coord['token_count']}")
    
    print()
    
    # 统计信息
    stats = tracker.stats()
    print(f"  统计信息:")
    print(f"    总消息数: {stats['total_messages']}")
    print(f"    总行数: {stats['total_lines']}")
    print(f"    总字符数: {stats['total_chars']}")
    print(f"    总 Token 数: {stats['total_tokens']}")
    
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ============================================================================
# 示例3：使用 ContextCompressor 压缩上下文
# ============================================================================
print("\n🗜️  示例3：智能上下文压缩")
print("-" * 70)

try:
    import hermes_context_engine as hce
    
    # 创建 compressor
    compressor = hce.PyContextCompressor("keep_recent", 2)
    
    # 准备消息列表
    messages = json.dumps([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
        {"role": "user", "content": "Tell me more."},
        {"role": "assistant", "content": "AI is a broad field..."},
        {"role": "user", "content": "Thanks!"},
    ])
    
    token_counts = [10, 5, 20, 5, 15, 3]
    
    # 压缩
    result = compressor.compress(messages, token_counts)
    
    print(f"  原始消息数: {result['original_count']}")
    print(f"  原始 Token 数: {result['original_tokens']}")
    print(f"  压缩后消息数: {result['compressed_count']}")
    print(f"  压缩后 Token 数: {result['compressed_tokens']}")
    print(f"  压缩比: {result['compression_ratio']:.1%}")
    print(f"  保留的消息 ID: {result['retained_message_ids']}")
    
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ============================================================================
# 示例4：使用多模型路由器
# ============================================================================
print("\n🔀 示例4：多模型路由器")
print("-" * 70)

try:
    import hermes_multi_model_router as hmr
    
    # 创建路由器（使用默认配置）
    router = hmr.PyRouter(None)
    print("  ✅ 路由器创建成功")
    
    # 设置并发模式
    modes = [
        ("single", None, "单模型模式"),
        ("top_n", 3, "并发前3个模型"),
        ("all", None, "并发所有模型"),
    ]
    
    for mode, param, desc in modes:
        router.set_concurrent_mode(mode, param)
        print(f"  ✅ 设置并发模式: {desc}")
    
    print()
    
    # 设置仲裁策略
    strategies = [
        ("first_success", "使用第一个成功的响应"),
        ("fastest", "使用最快的响应"),
        ("longest", "使用最长的响应"),
    ]
    
    for strategy, desc in strategies:
        router.set_arbitration_strategy(strategy)
        print(f"  ✅ 设置仲裁策略: {desc}")
    
    print()
    print("  ⚠️  注意: 实际调用需要配置 API 密钥和提供商")
    
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ============================================================================
# 示例5：查看轨迹统计
# ============================================================================
print("\n📈 示例5：轨迹统计")
print("-" * 70)

try:
    from agent.trace_hooks import get_trace_stats, export_traces
    
    stats = get_trace_stats()
    
    if stats:
        print(f"  总轨迹数: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        print(f"  平均耗时: {stats['avg_duration_ms']:.2f}ms")
        
        # 导出轨迹
        export_path = "/tmp/hermes_traces_example.json"
        export_traces(export_path)
        print(f"\n  ✅ 轨迹已导出到: {export_path}")
    else:
        print("  ℹ️  暂无轨迹数据（运行一些任务后会自动积累）")
    
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 所有示例执行完成！")
print("=" * 70)

print("\n📚 更多信息：")
print("  • 完整文档: ~/.hermes/core-reform/docs/final_complete_report.md")
print("  • 源代码: ~/.hermes/core-reform/")
print("  • 备份: ~/.hermes/backups/core_reform_20260525_014817/")

print("\n🚀 下一步：")
print("  1. 运行 Hermes 任务，轨迹会自动记录")
print("  2. 使用 Context Engine 替换现有 token 计数")
print("  3. 配置多模型路由器（可选）")
