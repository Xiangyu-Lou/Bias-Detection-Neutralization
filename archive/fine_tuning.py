from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-#####"

client = OpenAI()

# client.files.create(
#   file=open("data/fine_tuning_train_data.jsonl", "rb"),
#   purpose="fine-tune"
# )

client.fine_tuning.jobs.create(
  training_file="file-#####",
  model="gpt-3.5-turbo"
)