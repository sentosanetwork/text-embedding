from pydantic import BaseModel, Field
from typing import List

class TextsInput(BaseModel):
    texts: List[str] = Field(
        ...,
        example = [
            "query: how much protein should a female eat",
            "query: Công thức nấu ăn bí ngô tự làm",
            "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "passage: 1. Bí ngô xào sợi Nguyên liệu: nửa quả bí ngô mềm Gia vị: hành, muối, đường, cốt gà Cách làm: 1. Dùng dao gọt bỏ một lớp vỏ mỏng trên bề mặt bí ngô, dùng dao cạo sạch phần thịt thìa 2. Bào thành từng sợi mỏng (Nếu không có thớt thì dùng dao cắt từ từ thành từng sợi mỏng) 3. Đun nóng nồi, cho dầu vào, cho hành lá cắt nhỏ vào xào cho đến khi có mùi thơm 4. Thêm Bí ngô cắt nhỏ và xào nhanh trong khoảng một phút, thêm muối, một ít đường và nước cốt gà cho vừa ăn rồi dùng 2. Bí ngô xào hẹ Nguyên liệu: 1 quả bí ngô Gia vị: hẹ, tỏi băm, dầu ô liu, muối Cách làm: 1. Gọt vỏ bí ngô và cắt thành từng lát 2. Sau khi chảo dầu nóng 80%, cho tỏi băm vào xào cho đến khi có mùi thơm 3. Sau khi xào xong, cho các lát bí đỏ vào xào chín 4. Trong khi xào, bạn. Thỉnh thoảng có thể thêm nước vào nồi nhưng không quá nhiều 5. Thêm muối và xào đều 6. Bí đỏ gần mềm sau đó có thể tắt lửa 7. Rắc vào. hẹ và phục vụ."
        ]
    )

class EmbeddingsOutput(BaseModel):
    data: List[List[float]] = Field(
        ...,
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

class ScoresOutput(BaseModel):
    data: List[List[float]] = Field(
        ...,
        example=[[12.5, 34.2], [56.1, 78.9]]
    )
