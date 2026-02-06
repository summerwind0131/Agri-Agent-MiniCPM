from modelscope import snapshot_download
# 下载 MiniCPM-Llama3-V-2.5

# model_dir = snapshot_download('OpenBMB/MiniCPM-Llama3-V-2_5', cache_dir='./models')
# print("Model downloaded to:", model_dir)

# from modelscope import snapshot_download

# 修改 cache_dir 为数据盘路径
model_dir = snapshot_download('OpenBMB/MiniCPM-Llama3-V-2_5', cache_dir='/root/autodl-tmp/models')
print("Model downloaded to:", model_dir)