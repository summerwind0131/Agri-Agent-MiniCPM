import gradio as gr
import cv2
import time
import pandas as pd
import threading
from PIL import Image
from agri_agent import AgriAgent
from ui_utils import SmartHUD

# --- 1. 模型路径 ---
MODEL_PATH='/root/autodl-tmp/models/OpenBMB/MiniCPM-Llama3-V-2_5'
VIDEO_PATH = 'demo_video.mp4'
THINK_INTERVAL = 5

# --- 全局变量 ---
agent = None
hud = None
is_running = False

mock_sensors = {
    "battery": 98,
    "speed": 0.0,
    "lat": 34.0522,
    "lon": 118.2437
}

# --- 新增全局锁和共享数据 ---
data_lock = threading.Lock()
# 用于存储最新的 AI 决策结果
latest_inference_result = {
    "diagnosis": "等待数据...",
    "cmd": "待命"
}

def load_system():
    global agent, hud
    if agent is None:
        try:
            print("正在初始化 AI 大脑...")
            agent = AgriAgent(model_path=MODEL_PATH)
            agent.load_model() 
            hud = SmartHUD(font_path='SimHei.ttf')
            return "✅ 系统就绪 (System Online)"
        except Exception as e:
            return f"❌ 初始化失败: {str(e)}"
    return "系统已在运行 (System Already Online)"



# --- 后台线程函数：专门负责费时的 AI 计算 ---
def run_ai_background(img_pil):
    global latest_inference_result, agent, mock_sensors
    
    # 1. 耗时的预测过程
    raw_result = agent.predict(img_pil)
    key = raw_result.strip().replace(".", "")
    
    cn_map = {"Healthy": "健康", "Disease": "病害", "Pest": "虫害", "Unknown": "未知"}
    diagnosis = cn_map.get(key, key)
    
    # 2. 决策逻辑
    if "Healthy" in key:
        cmd = "全速巡航"
        target_speed = 1.5 
    elif "Disease" in key or "Pest" in key:
        cmd = "停车/喷洒"
        target_speed = 0.0
    else:
        cmd = "减速观察"
        target_speed = 0.5
        
    # 3. 安全地更新共享数据 (加锁)
    with data_lock:
        latest_inference_result["diagnosis"] = diagnosis
        latest_inference_result["cmd"] = cmd
        mock_sensors['speed'] = target_speed

def processing_loop():
    global is_running, mock_sensors, latest_inference_result
    
    if agent is None:
        yield None, "传感器离线", pd.DataFrame(), "⚠️ 请先点击 [1. 初始化系统]"
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    logs = []
    
    # 使用 try...finally 确保视频流一定会被关闭
    try:
        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_count += 1
            
            # --- 异步触发 AI (不等待结果) ---
            if frame_count % THINK_INTERVAL == 0 :
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 启动线程，daemon=True 表示主程序退出时线程自动结束
                threading.Thread(target=run_ai_background, args=(img_pil,), daemon=True).start()
            
            # --- 读取最新状态 (加锁读取) ---
            with data_lock:
                current_diagnosis = latest_inference_result["diagnosis"]
                current_cmd = latest_inference_result["cmd"]
                
                # 只有当有新结果或定期时更新日志，防止日志刷新太快
                if frame_count % THINK_INTERVAL == 0 :
                     timestamp = time.strftime("%H:%M:%S")
                     logs.insert(0, [timestamp, frame_count, current_diagnosis, current_cmd])
                     if len(logs) > 10: logs.pop()

            # --- 传感器模拟 ---
            mock_sensors['battery'] -= 0.01 
            if mock_sensors['battery'] < 0: mock_sensors['battery'] = 100
            
            # --- 绘制 HUD ---
            # 注意：这里的 state 不再需要显示 "思考中"，因为视频很流畅
            hud_data = {
                'frame_id': frame_count,
                'state': 'AI 监测中 (流畅模式)', 
                'diagnosis': current_diagnosis,
                'advice': f"目标速度: {mock_sensors['speed']} m/s",
                'cmd': current_cmd,
                'latency': 0
            }
            frame = hud.render_panel(frame, hud_data)
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            status_text = (
                f"🔋 电池: {int(mock_sensors['battery'])}%\n"
                f"🚀 速度: {mock_sensors['speed']} m/s\n"
                f"📍 坐标: ({mock_sensors['lat']}, {mock_sensors['lon']})"
            )
            
            log_df = pd.DataFrame(logs, columns=["时间", "帧号", "识别结果", "执行指令"])
            
            yield out_frame, status_text, log_df, "🟢 正在巡航 (Patrolling)"
            # time.sleep(0.003) # 控制帧率

    finally:
        cap.release()
        yield None, "已停止", pd.DataFrame(), "🔴 任务结束"
"""
def processing_loop():
    global is_running, mock_sensors
    
    if agent is None:
        # 如果没初始化，直接返回空
        yield None, "传感器离线", pd.DataFrame(), "⚠️ 请先点击 [1. 初始化系统]"
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    logs = []
    
    current_diagnosis = "等待数据..."
    current_cmd = "待命"
    
    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        
        # AI 推理
        if frame_count % THINK_INTERVAL == 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_result = agent.predict(img_pil)
            key = raw_result.strip().replace(".", "")
            
            cn_map = {"Healthy": "健康", "Disease": "病害", "Pest": "虫害", "Unknown": "未知"}
            current_diagnosis = cn_map.get(key, key)
            
            if "Healthy" in key:
                current_cmd = "全速巡航"
                mock_sensors['speed'] = 1.5 
            elif "Disease" in key or "Pest" in key:
                current_cmd = "停车/喷洒"
                mock_sensors['speed'] = 0.0
            else:
                current_cmd = "减速观察"
                mock_sensors['speed'] = 0.5
                
            timestamp = time.strftime("%H:%M:%S")
            logs.insert(0, [timestamp, frame_count, current_diagnosis, current_cmd])
            if len(logs) > 10: logs.pop()

        # 传感器模拟
        mock_sensors['battery'] -= 0.01 
        if mock_sensors['battery'] < 0: mock_sensors['battery'] = 100
        
        # 绘制 HUD
        hud_data = {
            'frame_id': frame_count,
            'state': 'AI 托管中' if frame_count % THINK_INTERVAL != 0 else '思考中...',
            'diagnosis': current_diagnosis,
            'advice': f"当前车速: {mock_sensors['speed']} m/s",
            'cmd': current_cmd,
            'latency': 0
        }
        frame = hud.render_panel(frame, hud_data)
        out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status_text = (
            f"🔋 电池: {int(mock_sensors['battery'])}%\n"
            f"🚀 速度: {mock_sensors['speed']} m/s\n"
            f"📍 坐标: ({mock_sensors['lat']}, {mock_sensors['lon']})"
        )
        
        log_df = pd.DataFrame(logs, columns=["时间", "帧号", "识别结果", "执行指令"])
        
        yield out_frame, status_text, log_df, "🟢 正在巡航 (Patrolling)"
        time.sleep(0.03)

    cap.release()
    yield None, "已停止", pd.DataFrame(), "🔴 任务结束"
"""
# --- 2. 生成器调用 ---
def start_patrol():
    global is_running
    is_running = True
    # 关键修改：直接把生成器的值一个个 yield 出来
    for output in processing_loop():
        yield output

def stop_patrol():
    global is_running
    is_running = False
    return "🔴 停止指令已发送"

with gr.Blocks(title="AgriAgent 指挥中心", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚜 AgriAgent 智能农业机器人 - 远程指挥中心")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_display = gr.Image(label="实时回传画面", type="numpy")
        
        with gr.Column(scale=1):
            system_status = gr.Textbox(label="系统日志", value="等待初始化...", interactive=False)
            sensor_display = gr.Textbox(label="传感器遥测", lines=3, interactive=False)
            
            init_btn = gr.Button("1. 初始化系统", variant="primary")
            start_btn = gr.Button("2. 开始巡航", variant="secondary")
            stop_btn = gr.Button("3. 紧急停止", variant="stop")

    gr.Markdown("### 📋 AI 诊断日志")
    log_table = gr.Dataframe(headers=["时间", "帧号", "识别结果", "执行指令"], interactive=False)

    init_btn.click(load_system, inputs=[], outputs=[system_status])
    
    start_btn.click(
        start_patrol, 
        inputs=[], 
        outputs=[video_display, sensor_display, log_table, system_status]
    )
    stop_btn.click(stop_patrol, inputs=[], outputs=[system_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)