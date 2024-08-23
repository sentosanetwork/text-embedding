import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: Công thức nấu ăn bí ngô tự làm',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: 1. Bí ngô xào sợi Nguyên liệu: nửa quả bí ngô mềm Gia vị: hành, muối, đường, cốt gà Cách làm: 1. Dùng dao gọt bỏ một lớp vỏ mỏng trên bề mặt bí ngô, dùng dao cạo sạch phần thịt thìa 2. Bào thành từng sợi mỏng (Nếu không có thớt thì dùng dao cắt từ từ thành từng sợi mỏng) 3. Đun nóng nồi, cho dầu vào, cho hành lá cắt nhỏ vào xào cho đến khi có mùi thơm 4. Thêm Bí ngô cắt nhỏ và xào nhanh trong khoảng một phút, thêm muối, một ít đường và nước cốt gà cho vừa ăn rồi dùng 2. Bí ngô xào hẹ Nguyên liệu: 1 quả bí ngô Gia vị: hẹ, tỏi băm, dầu ô liu, muối Cách làm: 1. Gọt vỏ bí ngô và cắt thành từng lát 2. Sau khi chảo dầu nóng 80%, cho tỏi băm vào xào cho đến khi có mùi thơm 3. Sau khi xào xong, cho các lát bí đỏ vào xào chín 4. Trong khi xào, bạn. Thỉnh thoảng có thể thêm nước vào nồi nhưng không quá nhiều 5. Thêm muối và xào đều 6. Bí đỏ gần mềm sau đó có thể tắt lửa 7. Rắc vào. hẹ và phục vụ."]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100

print(len(embeddings))
print(len(embeddings[0]))
print(embeddings)

print(scores)
print(scores.tolist())


# x = torch.rand(5, 3)
# print(x)
print(torch.cuda.is_available())