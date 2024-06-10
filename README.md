# Kendamarron/fineweb-edu-classifier-ja
https://huggingface.co/Kendamarron/fineweb-edu-classifier-ja

[HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)を再現するために、日本語データで[pkshatech/GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja)を学習したモデルです。

学習データは、[oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301)の日本語サブセットから抽出した16913個の文書に対して、[TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)のQ3_Kを使ってスコアリングしたものを使用しています。

詳細: [https://zenn.dev/kendama/articles/aba63f14f88e6e](https://zenn.dev/kendama/articles/aba63f14f88e6e)

## 使い方
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("Kendamarron/fineweb-edu-classifier-ja")
model = AutoModelForSequenceClassification.from_pretrained("Kendamarron/fineweb-edu-classifier-ja", num_labels=6, classifier_dropout=0.0, hidden_dropout_prob=0.0)
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class
text = "富士山は、日本で最も有名な山であり、日本全土にわたる広大な自然公園の一つです。高さは3,776メートルで、日本で最も高い山です。富士山は、東京都、静岡県、山梨県の3つの県にまたがっています。"
print(predict(text))
# >> 2
```
