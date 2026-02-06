import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

class AgriAgent:
    def __init__(self, model_path, device='cuda'):
        """
        初始化 Agent
        :param model_path: 模型权重的绝对路径
        :param device: 运行设备 ('cuda' 或 'cpu')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        加载模型到显存。
        注意：这个过程比较耗时，程序启动时调用一次即可。
        """
        print(f"Loading AgriAgent from {self.model_path} ...")
        try:
            # 自动判断精度：4090/3090 用 bf16，其他用 fp16
            dtype = torch.float16
            if self.device == 'cuda' and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            ).to(dtype=dtype)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            self.model = self.model.to(device=self.device)
            self.model.eval()
            print("✅ AgriAgent Brain Online! (模型加载完毕)")
            
        except Exception as e:
            print(f"❌ Critical Error loading model: {e}")
            raise e

    def predict(self, image_input, prompt=None):
        """
        核心推理接口
        :param image_input: PIL.Image 对象 (注意不是 OpenCV 的 numpy 数组)
        :param prompt: 给大模型的提示词
        :return: 模型生成的字符串结果
        """
        if self.model is None:
            return "Error: Model not loaded. Call load_model() first."

        try:
            # 确保图片是 RGB 模式
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")
            
            final_prompt = (
                "Identify the crop status in the image. Select ONE category from: [Healthy, Disease, Pest, Unknown]. Do not explain. Just output the category word."
            )

            msgs = [{'role': 'user', 'content': final_prompt}]
            
            # 这里的参数可以根据上一轮 demo 的效果微调
            res = self.model.chat(
                image=image_input,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True, 
                temperature=0.7
            )
            return res
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return "Error during inference."

# 单元测试代码：只有直接运行这个文件时才会执行
if __name__ == "__main__":
    # 请修改为你服务器上的真实路径
    MODEL_PATH = '/root/AgriAgent/models/OpenBMB/MiniCPM-Llama3-V-2_5'
    
    agent = AgriAgent(model_path=MODEL_PATH)
    agent.load_model()
    
    # 创建一个纯色图片测试一下通路
    test_img = Image.new('RGB', (224, 224), color='red')
    print("Testing prediction...")
    result = agent.predict(test_img, "图里有什么颜色？")
    print(f"Result: {result}")