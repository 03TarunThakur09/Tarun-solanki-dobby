## To install the required libraries for your project, run the following commands:

``` 
!pip install datasets transformers huggingface_hub
!apt-get install git-lfs 
```
### The code you have provided is used to load the IMDB dataset using the load_dataset function from the datasets library.

### To use this code, you will need to first install the datasets library by running !pip install datasets in your command line or notebook environment.

### Once you have installed the datasets library, you can import it and load the IMDB dataset by running the following code:

```
from datasets import load_dataset
imdb = load_dataset("imdb")
```
### This code will download and set up the DistilBERT tokenizer, which can be used to tokenize text input in preparation for input into a DistilBERT model. The distilbert-base-uncased model is a smaller, uncased version of the original DistilBERT model, and is often used as a more lightweight alternative for tasks where high computational efficiency is important.

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

### This code applies the preprocess_function to the small_train_dataset and small_test_dataset, tokenizing the input text using the DistilBERT tokenizer and truncating the sequences to the maximum sequence length of the tokenizer (by default, this is set to 512 tokens for the DistilBERT tokenizer).
### The resulting tokenized_train and tokenized_test objects contain the tokenized and preprocessed text inputs that can be fed into a DistilBERT model.

```
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)
```
## Train the model
#### Define DistilBERT as our base model:
```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
```
### The code you provided defines a compute_metrics function that computes the evaluation metrics for a classification task using the accuracy and f1 metrics from the datasets library.
```
import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}
```






