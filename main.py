from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="dialogue.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


trainer.train()

input_text = "dis moi qu'a tu fais de beau hier?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
generated_text_samples = model.generate(input_ids, max_length=50, num_return_sequences=3, do_sample=True)

for i, sample in enumerate(generated_text_samples):
    print(f"Sample {i + 1}: {tokenizer.decode(sample, skip_special_tokens=True)}")