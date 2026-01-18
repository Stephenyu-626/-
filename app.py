# Flask 主应用
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.doubao_api import doubao_api
from models.text_classify import text_classifier
from models.sentiment_analysis import sentiment_analyzer
from models.translator import translator
from utils.creative_features import creative_features
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """豆包API对话接口"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        conversation_history = data.get('history', None)
        
        if not message:
            return jsonify({"success": False, "error": "消息不能为空"}), 400
        
        result = doubao_api.chat(message, system_prompt, conversation_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """文本分类接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = text_classifier.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    """情感分析接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = sentiment_analyzer.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """机器翻译接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        direction = data.get('direction', 'zh2en')  # zh2en 或 en2zh
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = translator.translate(text, direction)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/keywords', methods=['POST'])
def keywords():
    """关键词提取接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        topK = data.get('topK', 5)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.extract_keywords(text, topK)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/summary', methods=['POST'])
def summary():
    """文本摘要接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 100)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.text_summary(text, max_length)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wordfreq', methods=['POST'])
def wordfreq():
    """词频统计接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        topN = data.get('topN', 10)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.word_frequency(text, topN)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics', methods=['POST'])
def statistics():
    """文本统计接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.text_statistics(text)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/similarity', methods=['POST'])
def similarity():
    """文本相似度接口"""
    try:
        data = request.get_json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({"error": "两个文本都不能为空"}), 400
        
        result = creative_features.text_similarity(text1, text2)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect_language', methods=['POST'])
def detect_language():
    """语言检测接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.detect_language(text)
        if "error" in result:
            return jsonify(result)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "services": {
            "doubao_api": "available",
            "text_classify": "available" if text_classifier.model else "unavailable",
            "sentiment_analysis": "available" if sentiment_analyzer.model else "unavailable",
            "translation": "available" if translator.encoder else "simplified"
        }
    })

if __name__ == '__main__':
    print(f"启动 Flask 服务器...")
    print(f"访问地址: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"内网访问: http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

