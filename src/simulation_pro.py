import cv2
import time
from PIL import Image
from agri_agent import AgriAgent
from ui_utils import SmartHUD  # 导入刚才写的 UI 模块

# --- 配置 ---
VIDEO_PATH = 'demo_video.mp4'
OUTPUT_PATH = 'output_pro_hud.mp4' # 输出文件名
#MODEL_PATH = '/root/AgriAgent/models/OpenBMB/MiniCPM-Llama3-V-2_5' # 确认路径
MODEL_PATH='/root/autodl-tmp/models/OpenBMB/MiniCPM-Llama3-V-2_5'

THINK_INTERVAL = 30 

# --- 简单的专家知识库 (Mock Expert System) ---
# 根据关键词匹配建议
KNOWLEDGE_BASE = {
    "Healthy": "作物生长状况良好，建议保持当前水肥管理。",
    "Disease": "检测到疑似病害！建议立即停车采样，并喷洒杀菌剂。",
    "Pest":    "检测到害虫活动！建议释放捕食螨或进行物理诱捕。",
    "Unknown": "目标不明确，请人工接管或靠近观察。"
}

def main():
    print("🤖 System Booting...")
    # 1. 初始化
    agent = AgriAgent(model_path=MODEL_PATH)
    agent.load_model()
    hud = SmartHUD(font_path='SimHei.ttf') # 初始化 HUD
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # --- 状态数据包 ---
    # 这个字典专门用来传给 HUD 画图
    system_data = {
        'frame_id': 0,
        'state': '初始化 (Initializing)',
        'diagnosis': '等待数据...',
        'advice': '系统启动中...',
        'cmd': '[等待指令]',
        'latency': 0
    }

    print("🚀 Simulation Started!")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        system_data['frame_id'] += 1
        
        # === 思考逻辑 ===
        if system_data['frame_id'] % THINK_INTERVAL == 0:
            system_data['state'] = '思考中 (AI Thinking...)'
            print(f"Frame {system_data['frame_id']}: AI Thinking...", end="")
            
            # 推理
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            raw_result = agent.predict(img_pil) # 得到英文结果 e.g. "Healthy"
            t1 = time.time()
            
            # 数据更新
            diagnosis_key = raw_result.strip().replace(".", "") # 清洗
            system_data['latency'] = (t1 - t0) * 1000
            
            # 简单的翻译映射
            cn_map = {
                "Healthy": "健康 (Healthy)",
                "Disease": "病害 (Disease)", 
                "Pest": "虫害 (Pest)", 
                "Unknown": "未知 (Unknown)"
            }
            system_data['diagnosis'] = cn_map.get(diagnosis_key, diagnosis_key)
            
            # 查知识库
            system_data['advice'] = KNOWLEDGE_BASE.get(diagnosis_key, "无特定建议")
            
            # 生成指令
            if "Healthy" in diagnosis_key:
                system_data['cmd'] = "[指令] 全速巡航"
            elif "Disease" in diagnosis_key or "Pest" in diagnosis_key:
                system_data['cmd'] = "[指令] 停车/喷洒"
            else:
                system_data['cmd'] = "[指令] 减速慢行"
                
            print(f" -> {system_data['diagnosis']}")
            
        else:
            system_data['state'] = '巡航中 (Scanning)'

        # === 渲染 HUD ===
        # 这一行代码替代了之前那一堆 cv2.putText
        frame = hud.render_panel(frame, system_data)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 完成！请下载 '{OUTPUT_PATH}' 查看中文仪表盘效果。")

if __name__ == "__main__":
    main()