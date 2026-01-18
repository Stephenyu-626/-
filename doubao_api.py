# 豆包 API 集成
import http.client
import json
from config import DOUBAO_API_URL, DOUBAO_API_KEY, DOUBAO_MODEL

class DoubaoAPI:
    def __init__(self):
        self.api_url = DOUBAO_API_URL
        self.api_key = DOUBAO_API_KEY
        self.model = DOUBAO_MODEL
        # 从URL中提取主机
        if "https://" in self.api_url:
            self.host = self.api_url.replace("https://", "").split("/")[0]
            self.path = "/" + "/".join(self.api_url.replace("https://", "").split("/")[1:])
        else:
            self.host = "ark.cn-beijing.volces.com"
            self.path = "/api/v3/chat/completions"
    
    def chat(self, message, system_prompt="You are a helpful assistant.", conversation_history=None):
        """
        调用豆包API进行对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            conversation_history: 对话历史记录
        """
        try:
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # 添加历史对话
            if conversation_history:
                messages.extend(conversation_history)
            
            # 添加当前消息
            messages.append({
                "role": "user",
                "content": message
            })
            
            # 构建请求体
            payload = json.dumps({
                "model": self.model,
                "messages": messages
            })
            
            # 设置请求头
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 发送请求
            conn = http.client.HTTPSConnection(self.host)
            conn.request("POST", self.path, payload, headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            # 解析响应
            response_data = json.loads(data.decode("utf-8"))
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "content": content,
                    "full_response": response_data
                }
            else:
                return {
                    "success": False,
                    "error": "API响应格式错误",
                    "response": response_data
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"API调用失败: {str(e)}"
            }
    
    def ask(self, question):
        """简单问答接口"""
        return self.chat(question)

# 全局实例
doubao_api = DoubaoAPI()

