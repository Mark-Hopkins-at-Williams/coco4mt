from paralleldata import coco_data
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import evaluate
import numpy as np
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppresses a transformers warning

def train(src, tgt, line_file, evaluation_split="validation"):
    """
    Trains the mbart-large-50-many-to-many-mmt checkpoint given a source
    language, target language, file to take sentences from, and evalutation
    split from the coco4mt data. Expects ISO 639-1 language codes
    (e.g.: "en") for src and tgt.

    Parameters
    ----------
    src : String
        the source language. Expects the standard two-letter code.
    tgt : String
        the target language. Expects the standard two-letter code.
    line_file : String
        the name of the file containing the indices of the sentences from the coco4mt data for training
    evaluation_split : String
        the split of the coco4mt data to be used ("train", "validation", or "test")
    """
    lines = set()
    with open(line_file) as reader:
        for line in reader:
            line = int(line.strip())
            lines.add(line)

    # The code3, code5, and langs_model dictionaries allow for standard language inputs.
    code3 = {
        "en": "eng",
        "de": "deu",
        "id": "ind",
        "ko": "kor",
        "fr": "fra",
        "my": "mya",
        "gu": "guj"
    }
    code5 = {
        "en": "en_XX",
        "de": "de_DE",
        "id": "id_ID",
        "ko": "ko_KR",
        "fr": "fr_XX",
        "my": "my_MM",
        "gu": "gu_IN"
    }
    langs_model = {
        "en": {"de": "eng-deu", "id": "eng-ind", "ko": "eng-kor", "fr": "eng-fra", "my": "eng-mya", "gu": "eng-guj"},
        "de": {"en": "deu-eng", "id": "deu-ind", "ko": "deu-kor", "fr": "deu-fra", "my": "deu-mya", "gu": "deu-guj"},
        "id": {"en": "ind-eng", "de": "ind-deu", "ko": "ind-kor", "fr": "ind-fra", "my": "ind-mya", "gu": "ind-guj"},
        "ko": {"en": "kor-eng", "de": "kor-deu", "id": "kor-ind", "fr": "kor-fra", "my": "kor-mya", "gu": "kor-guj"},
        "fr": {"en": "fra-eng", "de": "fra-deu", "id": "fra-ind", "ko": "fra-kor", "my": "fra-mya", "gu": "fra-guj"},
        "my": {"en": "mya-eng", "de": "mya-deu", "id": "mya-ind", "ko": "mya-kor", "fr": "mya-fra", "gu": "mya-guj"},
        "gu": {"en": "guj-eng", "de": "guj-deu", "id": "guj-ind", "ko": "guj-kor", "fr": "guj-fra", "my": "guj-mya"}
    }
    split_datasets = coco_data(src, tgt, lines)
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint, return_tensors="pt")
    tokenizer.src_lang = code5[src]
    tokenizer.tgt_lang = code5[tgt]
    max_length = 128

    def preprocess_function(examples):
        """
        Enables the tokenizer.
        """
        inputs = [ex[code3[src]] for ex in examples["translation"]]
        targets = [ex[code3[tgt]] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=split_datasets["train"].column_names,
    )
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        """
        Sets up and evaluates with BLEU given a list of predictions.

        Parameters
        ----------
        eval_preds : list[list[String]]
            the model-generated translation predictions

        Returns
        -------
        dict
            shows the BLEU score
        """
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    prefix_fields = line_file.split(".")
    prefix = '.'.join(prefix_fields[:-1])
    model_dir = f'{langs_model[src][tgt]}-{prefix}'
    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[evaluation_split],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=(evaluation_split=="test"))
    print(f"After training ({model_dir}):")
    results = trainer.evaluate(max_length=max_length)
    print(results)
    with open(f"bleu.{evaluation_split}.{model_dir}", "w") as writer:
        writer.write(f"{results['eval_bleu']}\n")
    if evaluation_split == "validation":
        trainer.push_to_hub(tags="translation", commit_message="Training complete")


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])