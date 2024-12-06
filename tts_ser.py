import os
import re
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Seq2SeqTrainer, Seq2SeqTrainingArguments
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import pandas as pd

# from datasets import load_dataset, Audio
# https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered
dataset = load_dataset("DavronSherbaev/uzbekvoice-filtered", split="train")
dataset

# half_size = len(dataset) // 16

# # Select the first half of the dataset
# dataset = dataset.select(range(half_size))

# print(dataset)

dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint,token="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")

tokenizer = processor.tokenizer

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

# import re

# Function to convert numbers to words in Uzbek
def number_to_words(n):
    ones = ['nol', 'bir', 'ikki', 'uch', 'tört', 'besh', 'olti', 'yetti', 'sakkiz', 'töqqiz']
    tens = ['', 'ön', 'yigirma', 'öttiz', 'qirq', 'ellik', 'oltmish', 'yetmish', 'sakson', 'toqson']
    hundreds = ['', 'yuz', 'ikki yuz', 'uch yuz', 'tört yuz', 'besh yuz', 'olti yuz', 'yetti yuz', 'sakkiz yuz', 'töqqiz yuz']
    thousands = ['', 'ming', 'million', 'milliard', 'trillion']

    if n < 10:
        return ones[n]
    elif n < 100:
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
    elif n < 1000:
        return hundreds[n // 100] + (" " + number_to_words(n % 100) if n % 100 != 0 else "")
    elif n < 10**6:
        return number_to_words(n // 1000) + " ming" + (" " + number_to_words(n % 1000) if n % 1000 != 0 else "")
    elif n < 10**9:
        return number_to_words(n // 10**6) + " million" + (" " + number_to_words(n % 10**6) if n % 10**6 != 0 else "")
    elif n < 10**12:
        return number_to_words(n // 10**9) + " milliard" + (" " + number_to_words(n % 10**9) if n % 10**9 != 0 else "")
    else:
        return number_to_words(n // 10**12) + " trillion" + (" " + number_to_words(n % 10**12) if n % 10**12 != 0 else "")

# Function to normalize text (convert to lowercase, remove extra whitespace, and convert numbers)
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()

    # Convert numbers to words
    words = text.split()
    new_words = []
    for word in words:
        if word.isdigit():
            num = int(word)
            new_words.append(number_to_words(num))
        else:
            new_words.append(word)

    # Remove extra whitespace
    normalized_text = ' '.join(new_words)

    return normalized_text

# Define a function to add the normalized_text column
def add_normalized_text(example):
    example['normalized_text'] = normalize_text(example['text'])
    return example

# Apply the function to the dataset
dataset = dataset.map(add_normalized_text)

# Print the first few examples to verify
print(dataset[2:5])

def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

dataset_vocab - tokenizer_vocab

replacements = [
      ('г', 'g'),
      ('ⓣ', ''),
      ('/', ''),
      ('!', ''),
      ('«', ''),
      (':', ''),
      ('б', 'b'),
      ('о', 'o'),
      ("в", 'v'),
      (')', ''),
      ('ю', 'yu'),
      ('-', ' '),
      ('–', ' '),
      ('—', ' '),
      ('…', ''),
      ('л', 'l'),
      ('(', ''),
      ('д', 'd'),
      ('м', 'm'),
      ('ё', 'yo'),
      ('№', ''),
      ('у', 'u'),
      ("o'", 'ö'),
      ("g'", 'ğ'),
      ('е', 'e'),
      ('а', 'a'),
      ('р', 'r'),
      ('с', 's'),
      ('+', 'pilus'),
      ('?', ''),
      ('–', ''),
      ('х', 'x'),
      ('`', ''),
      ('-', ''),
      ('т', 't'),
      ("”", ''),
      ('»', ''),
      ('—', ''),
      ("'", ''),
      ('0', 'nol'),
      ('1', 'bir'),
      ('2', 'ikki'),
      ('3', 'uch'),
      ('4', 'tört'),
      ('5', 'besh'),
      ('6', 'olti'),
      ('7', 'yetti'),
      ('8', 'sakkiz'),
      ('9', 'töqqiz'),
      ('ж', 'j'),
      ('.', ' ')

]

def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs

dataset = dataset.map(cleanup_text)

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    audio = example["path"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())

processed_example["speaker_embeddings"].shape

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# lengths = [len(x["input_ids"]) for x in dataset]
# print(f"Average length: {sum(lengths) / len(lengths)}, Max length: {max(lengths)}")

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length <= 600
dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
len(dataset)

dataset = dataset.train_test_split(test_size=0.2)

# from dataclasses import dataclass
# from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint,token="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")

# from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)

# training_args = Seq2SeqTrainingArguments(
#     output_dir="ilhom_tts_version_1",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=16,
#     learning_rate=5e-5,
#     warmup_steps=200,
#     max_steps=2000,
#     gradient_checkpointing=True,
#     fp16=True,
#     eval_strategy="steps",  # Make this match the save strategy
#     save_strategy="steps",  # Match with eval_strategy
#     per_device_eval_batch_size=2,
#     save_steps=500,
#     eval_steps=500,
#     logging_steps=50,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     greater_is_better=False,
#     label_names=["labels"],
#     push_to_hub=False,
# )

training_args = Seq2SeqTrainingArguments(
    output_dir="ilhom_tts",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    per_device_eval_batch_size=4,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
      args=training_args,
      model=model,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      data_collator=data_collator,
      tokenizer=processor,
  )

dataset['train'].shape

trainer.train()

print("Bo'ldi Tugadi ...")

