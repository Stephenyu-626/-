#!/usr/bin/env python3
# 快速测试脚本 - 测试所有API接口

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_chat():
    """测试智能问答接口"""
    print("=" * 50)
    print("测试智能问答接口...")
    try:
        data = {
            "message": "你好，请介绍一下你自己"
        }
        response = requests.post(f"{BASE_URL}/api/chat", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_classify():
    """测试文本分类接口"""
    print("=" * 50)
    print("测试文本分类接口...")
    try:
        data = {
            "text": "今天股市大涨，投资者情绪高涨"
        }
        response = requests.post(f"{BASE_URL}/api/classify", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_sentiment():
    """测试情感分析接口"""
    print("=" * 50)
    print("测试情感分析接口...")
    try:
        data = {
            "text": "这个产品非常好用，我很满意！"
        }
        response = requests.post(f"{BASE_URL}/api/sentiment", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_translate():
    """测试机器翻译接口"""
    print("=" * 50)
    print("测试机器翻译接口...")
    try:
        data = {
            "text": "你好，世界",
            "direction": "zh2en"
        }
        response = requests.post(f"{BASE_URL}/api/translate", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_keywords():
    """测试关键词提取接口"""
    print("=" * 50)
    print("测试关键词提取接口...")
    try:
        data = {
            "text": "人工智能是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "topK": 5
        }
        response = requests.post(f"{BASE_URL}/api/keywords", json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("智能问答系统 API 测试")
    print("=" * 50)
    
    tests = [
        ("健康检查", test_health),
        ("智能问答", test_chat),
        ("文本分类", test_classify),
        ("情感分析", test_sentiment),
        ("机器翻译", test_translate),
        ("关键词提取", test_keywords),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{name} 测试失败: {e}")
            results.append((name, False))
        print()
    
    # 打印测试总结
    print("=" * 50)
    print("测试总结:")
    print("=" * 50)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")

if __name__ == "__main__":
    main()

