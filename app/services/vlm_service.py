"""VLM Image Service — DeepSeek Vision API for exercise image description."""
import base64
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

VLM_CLASSIFY_PROMPT = """分析这张图片的内容类型，只回复一个字母：
A - 纯文字/公式内容（如一道完整的习题截图）
B - 纯插图（如图形、图表、几何图）
C - 习题文字+插图混合"""

VLM_DESCRIBE_PROMPT = """详细描述这张图片中的数学内容。如果是公式请转为LaTeX。如果是几何图请描述图形特征。
如果看不清楚或无法识别，请如实说明。"""

VLM_TRANSCRIBE_PROMPT = """你是数学题目图片识别助手。请完整识别这张图片中的内容，转成文字 + LaTeX 公式。

要求：
1. 完整转录题目文字，不要遗漏任何数字、符号、上下标、分数、根式、矩阵、行列式。
2. 数学公式一律转为 LaTeX，行内公式用 $...$，独立（居中）公式用 $$...$$。
3. 如果图片里有图形（函数图像、几何图、表格、流程图、坐标图等），请详细描述：
   - 图形类型与关键特征（坐标轴范围、曲线形状、交点/切点/极值点等）
   - 图中标注的字母、数字、坐标、角度、边长
   - 图形之间的几何关系（平行、垂直、相切、相似、对称等）
4. 直接输出识别结果，不要加"这是"、"图片内容是"之类的前缀或解释。"""


class VLMService:
    """Image analysis using DeepSeek Vision API."""

    def __init__(self):
        self.base_url = settings.VLM_BASE_URL or settings.DEEPSEEK_BASE_URL
        self.model = settings.VLM_MODEL or settings.DEEPSEEK_CHAT_MODEL
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=self.base_url,
        )

    def _encode_image(self, image_path: str) -> str:
        """Read image file and encode to base64 data URL."""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower().replace(".", "")
        mime = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "gif", "webp") else "image/png"
        return f"data:{mime};base64,{data}"

    def classify(self, image_path: str) -> str:
        """Classify image content type: text_exercise / illustration / mixed.

        Returns: "text_exercise" | "illustration" | "mixed"
        """
        try:
            data_url = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_CLASSIFY_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=10,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().upper()
            if answer.startswith("A"):
                return "text_exercise"
            elif answer.startswith("C"):
                return "mixed"
            return "illustration"
        except Exception as e:
            logger.warning("VLM classify failed: %s", e)
            return "illustration"

    def describe(self, image_path: str) -> Optional[str]:
        """Generate a text description of an image for embedding."""
        try:
            data_url = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_DESCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("VLM describe failed: %s", e)
            return None

    def transcribe_image(self, image_path: str) -> Optional[str]:
        """Recognize text + math formulas in an image, output as text + LaTeX.

        Returns the transcribed text, or None if recognition failed.
        """
        try:
            data_url = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_TRANSCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=3000,
                temperature=0.1,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as e:
            logger.warning("VLM transcribe failed: %s", e)
            return None

    def quality_check(self, description: str) -> bool:
        """Quick heuristic: is the VLM description useful for retrieval?

        Returns True if the description has enough math/structural content.
        """
        if not description or len(description) < 20:
            return False
        # Has LaTeX or math notation
        if "$" in description or "\\\\" in description:
            return True
        # Has descriptive math terms
        math_terms = ["函数", "曲线", "三角形", "向量", "矩阵", "函数图",
                       "坐标", "角度", "积分", "function", "graph", "curve"]
        if any(t in description for t in math_terms):
            return True
        return False

    def process_image(self, image_path: str) -> dict:
        """Full pipeline: classify → maybe describe → quality check.

        Returns: {"type": "text_exercise|illustration|mixed",
                   "description": "...", "usable": bool}
        """
        img_type = self.classify(image_path)
        result = {"type": img_type, "description": None, "usable": False}

        if img_type == "text_exercise":
            # Text exercise: describe for OCR-like content
            desc = self.describe(image_path)
            result["description"] = desc
            result["usable"] = self.quality_check(desc or "")
        elif img_type == "mixed":
            # Mixed: describe for retrieval enhancement
            desc = self.describe(image_path)
            result["description"] = desc
            result["usable"] = self.quality_check(desc or "")
        else:
            # Pure illustration: describe, may be useful
            desc = self.describe(image_path)
            result["description"] = desc
            result["usable"] = self.quality_check(desc or "")

        return result
