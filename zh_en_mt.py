from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# tokenizer = AutoTokenizer.from_pretrained("../opus-mt-zh-en")
#
# model = AutoModelForSeq2SeqLM.from_pretrained("../opus-mt-zh-en")
#
# text = "从时间上看，中国空间站的建造比国际空间站晚20多年。"
# # Tokenize the text
# batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])
#
# # Make sure that the tokenized text does not exceed the maximum
# # allowed size of 512
# batch["input_ids"] = batch["input_ids"][:, :512]
# batch["attention_mask"] = batch["attention_mask"][:, :512]
#
# # Perform the translation and decode the output
# translation = model.generate(**batch)
# result = tokenizer.batch_decode(translation, skip_special_tokens=True)
# print(result)



# 加载zh-to-en模型

tokenizer = AutoTokenizer.from_pretrained("../opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("../opus-mt-zh-en")
# 加载en-to-zh模型
# tokenizer_back_translate = AutoTokenizer.from_pretrained("../opus-mt-zh-en")
# model_back_translate = AutoModelForSeq2SeqLM.from_pretrained("../opus-mt-zh-en")
# 创建zh2en和en2zh管道
zh2en = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
# en2zh = pipeline("translation_en_to_zh", model=model_back_translate, tokenizer=tokenizer_back_translate)

texts = ["打开沙发旁边墙上的灯","打开沙发旁边柜子上的灯","打开柜子里的灯","打开餐桌边上墙边的灯","打开餐桌旁边靠墙的灯","打开餐餐厅桌子旁边靠墙的灯"]
# 将输入文本翻译成英文
for text in texts:
    text_en = zh2en(text, max_length=510)[0]["translation_text"]
    print(text,":\t", text_en)
# 将英文翻译回中文
# text_back = self.en2zh(text_en, max_length=510)[0]["translation_text"]
# print("text_back:", text_back)