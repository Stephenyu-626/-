# 创意功能模块
import re
import jieba
import jieba.analyse
from collections import Counter
from difflib import SequenceMatcher
import math

class CreativeFeatures:
    @staticmethod
    def extract_keywords(text, topK=5):
        """提取关键词（基于 jieba TF-IDF），返回按权重排序的前 topK 个关键词。"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "待提取文本不能为空"}
            # 入参保护
            try:
                topK = int(topK)
            except Exception:
                topK = 5
            topK = max(1, min(topK, 50))

            # 使用 TF-IDF 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=True)
            return {
                "keywords": [{"word": word, "weight": float(weight)} for word, weight in keywords],
                "top_keywords": [word for word, _ in keywords]
            }
        except Exception as e:
            return {"error": f"关键词提取失败: {str(e)}"}
    
    @staticmethod
    def text_summary(text, max_length=100):
        """文本摘要（简化版，基于句子截断），适合快速体验。"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "待摘要文本不能为空"}
            try:
                max_length = int(max_length)
            except Exception:
                max_length = 100
            max_length = max(30, min(max_length, 500))

            if len(text) <= max_length:
                return {"summary": text, "original_length": len(text), "summary_length": len(text)}
            
            # 简单摘要：按中英文标点切分，取前几句和后几句
            sentences = re.split(r'[。！？!?\n]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 2:
                summary = text[:max_length] + "..."
            else:
                # 取前2句和后1句
                summary = "。".join(sentences[:2])
                if len(sentences) > 3:
                    summary += "。" + sentences[-1]
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text), 2)
            }
        except Exception as e:
            return {"error": f"文本摘要失败: {str(e)}"}
    
    @staticmethod
    def word_frequency(text, topN=10):
        """词频统计"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "待统计文本不能为空"}
            try:
                topN = int(topN)
            except Exception:
                topN = 10
            topN = max(1, min(topN, 100))

            words = list(jieba.cut(text))
            # 过滤停用词和标点
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                         '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                         '自己', '这', '，', '。', '？', '！', '、', '；', '：', '"', "'", '（', '）'}
            words = [w for w in words if w.strip() and w not in stop_words and len(w) > 1]
            
            word_freq = Counter(words)
            top_words = word_freq.most_common(topN)
            
            return {
                "word_frequency": [{"word": word, "count": count} for word, count in top_words],
                "total_words": len(words),
                "unique_words": len(word_freq)
            }
        except Exception as e:
            return {"error": f"词频统计失败: {str(e)}"}
    
    @staticmethod
    def text_statistics(text):
        """文本统计信息"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "待统计文本不能为空"}

            stats = {
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(' ', '')),
                "word_count": len(list(jieba.cut(text))),
                "sentence_count": len(re.split(r'[。！？\n]', text)),
                "paragraph_count": len([p for p in text.split('\n') if p.strip()]),
                "avg_sentence_length": 0,
                "avg_word_length": 0
            }
            
            sentences = [s for s in re.split(r'[。！？\n]', text) if s.strip()]
            if sentences:
                stats["avg_sentence_length"] = round(sum(len(s) for s in sentences) / len(sentences), 2)
            
            words = list(jieba.cut(text))
            if words:
                stats["avg_word_length"] = round(sum(len(w) for w in words) / len(words), 2)
            
            return stats
        except Exception as e:
            return {"error": f"文本统计失败: {str(e)}"}
    
    @staticmethod
    def detect_language(text):
        """检测语言（简化版）"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "待检测文本不能为空"}

            # 简单检测：中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            total_chars = len(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]', text))
            
            if total_chars == 0:
                return {"language": "未知", "confidence": 0}
            
            chinese_ratio = chinese_chars / total_chars
            english_ratio = english_chars / total_chars
            
            if chinese_ratio > 0.5:
                return {"language": "中文", "confidence": chinese_ratio}
            elif english_ratio > 0.5:
                return {"language": "英文", "confidence": english_ratio}
            else:
                return {"language": "混合", "confidence": 0.5}
        except Exception as e:
            return {"error": f"语言检测失败: {str(e)}"}

    @staticmethod
    def text_similarity(text1, text2):
        """文本相似度（字符/Jaccard/余弦综合）"""
        try:
            if not isinstance(text1, str) or not text1.strip() or not isinstance(text2, str) or not text2.strip():
                return {"error": "两个文本都不能为空"}

            # 字符相似度
            char_sim = SequenceMatcher(None, text1, text2).ratio()

            # 分词
            words1 = [w for w in jieba.lcut(text1) if w.strip()]
            words2 = [w for w in jieba.lcut(text2) if w.strip()]
            set1, set2 = set(words1), set(words2)

            union = len(set1 | set2)
            inter = len(set1 & set2)
            jaccard = inter / union if union else 0.0

            # 余弦
            v1, v2 = Counter(words1), Counter(words2)
            all_words = set(v1.keys()) | set(v2.keys())
            dot = sum(v1[w] * v2[w] for w in all_words)
            norm1 = math.sqrt(sum(v1[w] ** 2 for w in all_words))
            norm2 = math.sqrt(sum(v2[w] ** 2 for w in all_words))
            cosine = dot / (norm1 * norm2) if norm1 and norm2 else 0.0

            overall = (char_sim + jaccard + cosine) / 3
            to_pct = lambda x: round(x * 100, 2)
            return {
                "similarity": to_pct(overall),
                "char_similarity": to_pct(char_sim),
                "jaccard_similarity": to_pct(jaccard),
                "cosine_similarity": to_pct(cosine)
            }
        except Exception as e:
            return {"error": f"相似度计算失败: {str(e)}"}

# 全局实例
creative_features = CreativeFeatures()

