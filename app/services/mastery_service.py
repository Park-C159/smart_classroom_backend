"""知识点掌握度更新助手（学情分析）。

掌握度用指数滑动平均（EWMA）更新：new = old * 0.7 + rate * 0.3，
初始值 0.5，结果钳制在 [0.1, 1.0]。作业/测试/考试提交时按逐题知识点调用。
"""
from sqlalchemy import select

from app.models import KPMastery

MASTERY_FLOOR = 0.1
MASTERY_CEIL = 1.0
EWMA_WEIGHT = 0.3  # new = old * (1-weight) + rate * weight
INITIAL_MASTERY = 0.5


def _clamp(v: float) -> float:
    return max(MASTERY_FLOOR, min(MASTERY_CEIL, v))


async def update_kp_mastery(db, user_id: int, kp_id: str, correct_rate: float) -> None:
    """按 EWMA 更新某个知识点的掌握度。

    Args:
        db: AsyncSession
        user_id: 用户 ID
        kp_id: 知识点 ID（空则跳过）
        correct_rate: 本次该知识点的正确率（0-1）
    """
    if not kp_id:
        return
    rate = max(0.0, min(1.0, correct_rate))

    result = await db.execute(
        select(KPMastery).where(
            KPMastery.user_id == user_id,
            KPMastery.kp_id == kp_id,
        )
    )
    m = result.scalar_one_or_none()
    if m:
        m.mastery = round(_clamp(m.mastery * (1 - EWMA_WEIGHT) + rate * EWMA_WEIGHT), 3)
        m.total_questions += 1
    else:
        db.add(KPMastery(
            user_id=user_id,
            kp_id=kp_id,
            mastery=round(_clamp(INITIAL_MASTERY * (1 - EWMA_WEIGHT) + rate * EWMA_WEIGHT), 3),
            total_questions=1,
        ))
