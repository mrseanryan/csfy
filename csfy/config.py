# KEEP THIS FILE IN SYNC WITH CONFIG.INI
### ========= train command =========
EPOCHS = 3
LEARNING_RATE = 2e-5

# Edit the column names to match your data
COLUMN_TEXT = 'text'
COLUMN_LABEL = 'label'

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
WEIGHT_DECAY = 0.01

TRUNCATE_DATA_ROWS = None # 100 set to a number to truncate the rows of data used for training (for quick testing)

### ========= export command =========
EXAMPLE_USER_PROMPT = "How do I get to the beach?"  # edit this to match your data: for example, matching one text value

### ========= shared =========
BASE_MODEL = 'distilbert-base-uncased'
TOKENIZER = 'distilbert-base-uncased'

OUTPUT_DIR = './models'
