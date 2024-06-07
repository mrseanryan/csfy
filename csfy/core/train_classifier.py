import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from cornsnake import util_dir, util_input, util_json, util_print

from csfy import config, util_config
from . import util_labels

def _save_results(results):
    file_path = os.path.join(util_config.path_to_results(), 'results.json')
    util_json.write_to_json_file(results, file_path)

def train(path_to_input_parquet):
    if os.path.exists(util_config.path_to_results()):
        util_print.print_warning("Results directory already exists - safer to adjust 'OUTPUT_DIR' in config.ini")
        do_continue = util_input.input_with_format_y_or_n("Do you wish to continue?", "Y")
        if not do_continue:
            print("(exiting)")
            return

    util_print.print_section("Reading input parquet")
    df = pd.read_parquet(path_to_input_parquet)
    print(f"{len(df)} rows of data")
    if config.TRUNCATE_DATA_ROWS:
        util_print.print_warning(f" - truncated to {config.TRUNCATE_DATA_ROWS} rows")
        df = df.sample(config.TRUNCATE_DATA_ROWS, random_state=42)

    df[config.COLUMN_TEXT] = df[config.COLUMN_TEXT].str.lower()

    label_encoder = LabelEncoder()
    df[config.COLUMN_LABEL] = label_encoder.fit_transform(df[config.COLUMN_LABEL])
    num_labels = len(label_encoder.classes_)

    mapping = util_labels.get_label_mapping(label_encoder)
    util_dir.ensure_dir_exists(util_config.path_to_results())
    util_labels.save_label_mapping(mapping)

    tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER)

    def tokenize_function(examples):
        return tokenizer(examples[config.COLUMN_TEXT], padding="max_length", truncation=True)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_function, batched=True)

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.shuffle(seed=42).select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    model = DistilBertForSequenceClassification.from_pretrained(config.BASE_MODEL, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=util_config.path_to_results(),
        evaluation_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    util_print.print_section("Training")
    trainer.train()

    # if small data, then can happen there were not enough steps to trigger an auto-save (see Trainer code)
    util_print.print_result(f"Saving model to {util_config.path_to_generated_model()}")
    trainer.save_model(util_config.path_to_generated_model())

    util_print.print_section("Evaluating")
    results = trainer.evaluate()

    print(results)
    util_print.print_result(f"Results are at {util_config.path_to_results()}")
    _save_results(results)
