import http.client
import json

conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")
payload = json.dumps({
   "model": "doubao-seed-1-6-250615",
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": "端口映射是什么意思？"
      }
   ]
})
headers = {
   'Authorization': 'Bearer 48a29225-a258-471c-97e6-4e1ebef8ae35',
   'Content-Type': 'application/json'
}
conn.request("POST", "/api/v3/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))