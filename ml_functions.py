import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class DescriptionDataset(Dataset):

  def __init__(self, uuids, descriptions, labels, tokenizer, max_len):
    self.uuid = uuids
    self.descriptions = descriptions
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.descriptions)

  def __getitem__(self, item):
    uuid = self.uuid[item]
    description = str(self.descriptions[item])
    label = self.labels[item]

    encoding = self.tokenizer.encode_plus(
      description,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )

    return {
      'uuid': uuid,
      'description': description,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'label': torch.tensor(label, dtype=torch.long)
    }


def create_data_loader(df, tokenizer=tokenizer, max_len=256, batch_size=16):
  ds = DescriptionDataset(
    uuids=df.uuid.to_numpy(),
    descriptions=df.description.to_numpy(),
    labels=df['is_sustainable'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )

def preprocess(data):
    df_dict = {'uuid': ['0'], 'description': [data], 'is_sustainable': [False]}
    df = pd.DataFrame(df_dict)

    return create_data_loader(df)


def evaluate(data, model):
    data_loader = preprocess(data)
    predictions = []
    prediction_probs = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to('cpu')
            attention_mask = d["attention_mask"].to('cpu')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    print('>>>', data, predictions)
    return postprocess(predictions)


def postprocess(predictions):
    print(predictions)
    if predictions[0] == 1:
        return True
    else:
        return False

class GreenClassifier(nn.Module):

  def __init__(self, n_classes=2):
    super(GreenClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(bert_output['pooler_output'])
    return self.out(output)

def load_model(model_path='model/bert128_10e_cat_long_desc.bin'):
    try:
        map_location = torch.device('cpu')  # Use CPU regardless of CUDA availability
        state_dict = torch.load(model_path, map_location=map_location)

        if 'module' in state_dict:
            state_dict = state_dict['module']

        model = GreenClassifier()  # Replace YourModelClass with the actual class of your model

        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

        model.load_state_dict(state_dict)

        model.eval()

        return model

    except Exception as e:
        raise Exception(f"Error loading the model: {str(e)}")

