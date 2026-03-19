import torch
import torch.nn as nn
from collections import defaultdict
from os.path import join
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from geneformer import DataCollatorForCellClassification
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evaluation.eval import eval_classifier, plot_classifier
import gc
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.logs_ import get_logger
logger = get_logger()

class SubsampledPatientDataset(TorchDataset):
    def __init__(self, data: HFDataset, max_cells_per_bag=500, max_cells_per_patient= 1000, max_number_genes=512):
        """
        Args:
            data: Hugging Face dataset with 'sample_id', 'input_ids', 'attention_mask'
            max_cells_per_bag: max number of cells per bag
        """
        self.max_cells = max_cells_per_bag
        self.bags = []  # List of dicts: each represents a bag
        self.data = data

        sample_ids = list(set(data["sample_id"]))
        for sample_id in sample_ids:
            patient_data = data.filter(lambda x: x["sample_id"] == sample_id)

            # Create custom mask as well (can be same as attention_mask or based on 'length')
            lengths = patient_data["length"]
            mask = [[1]*min(l, max_number_genes) + [0]*max(0, max_number_genes - l) for l in lengths]
            
            # Truncate or pad each 'input_ids' and 'attention_mask' to max_number_genes
            def pad_or_truncate(seq, max_len):
                return seq[:max_len] + [0] * max(0, max_len - len(seq))

            input_ids = [pad_or_truncate(seq, max_number_genes) for seq in patient_data["input_ids"]]
            # attention_mask = [pad_or_truncate(seq, max_number_genes) for seq in patient_data["attention_mask"]]

           

            # Add/replace columns
            patient_data = patient_data.remove_columns(["input_ids"])
            patient_data = patient_data.add_column("input_ids", input_ids)
            
            patient_data = patient_data.add_column("attention_mask", mask)
            
            num_cells = len(patient_data)
            if max_cells_per_patient>0:
                num_cells = min(num_cells, max_cells_per_patient)
            num_bags = (num_cells + max_cells_per_bag - 1) // max_cells_per_bag

            indices = list(range(num_cells))

            for i in range(num_bags):
                start = i * max_cells_per_bag
                end = min((i + 1) * max_cells_per_bag, num_cells)
                bag_indices = indices[start:end]

                self.bags.append({
                    "input_ids": [patient_data[i]["input_ids"] for i in bag_indices],
                    "attention_mask": [patient_data[i]["attention_mask"] for i in bag_indices],
                    "label": patient_data[0]["label"],  # same for all bags of this patient
                    "sample_id": sample_id
                })

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        ret = {
            "input_ids": torch.tensor(bag["input_ids"]),
            "attention_mask": torch.tensor(bag["attention_mask"]),
            "label": torch.tensor(bag["label"], dtype=torch.float),
            "bag_size": len(bag["input_ids"]),
            "sample_id": bag['sample_id']
        }
        print(bag['sample_id'])
        return ret

def collate_patient_level(batch):
    """
    batch: list of dicts returned from PatientDataset.__getitem__
    """
    input_ids, attention_masks, labels, bag_sizes, sample_ids = [], [], [], [], []

    for patient in batch:
        # print(patient)
        # ds = patient["dataset"]  # Hugging Face dataset
        ds = patient
        input_ids.extend(ds["input_ids"])
        attention_masks.extend(ds["attention_mask"])  # assuming this is attention_mask
        bag_sizes.append(len(ds))
        labels.append(ds["label"])  # all cells should share same label
        sample_ids.append(ds["sample_id"])

    # Pad cell sequences
    input_ids = pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True)
    attention_masks = pad_sequence([torch.tensor(x) for x in attention_masks], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float)
    bag_sizes = torch.tensor(bag_sizes)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "bag_sizes": bag_sizes,
        "sample_ids" : sample_ids
    }

class ChunkedBertMILClassifier_check(nn.Module):
    def __init__(self, pretrained_model_path="bert-base-uncased", num_labels=1, chunk_size=128, device='cpu'):
        super().__init__()
        self.chunk_size = chunk_size
        self.device = device

        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

        # TODO: check if this is needed. Enable gradient checkpointing to reduce memory
        self.bert.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, bag_sizes=None):
        """
        Assumes batch_size = 1 (one patient), so all input_ids belong to one patient.
        input_ids: [num_cells, seq_len]
        attention_mask: [num_cells, seq_len]
        Returns:
            - patient_logits: [1]
        """
        chunk_size = self.chunk_size
        device = self.device

        logits_sum = None
        total_cells = input_ids.size(0)

        for i in range(0, total_cells, chunk_size):
            # print(f'cell {i+1}-{(i+1)*chunk_size} of {total_cells} per bag')

            chunk_input_ids = input_ids[i:i + chunk_size].to(device)
            chunk_attention_mask = attention_mask[i:i + chunk_size].to(device)

            # Forward with gradient tracking
            chunk_logits = self.bert(chunk_input_ids,
                                     attention_mask=chunk_attention_mask,
                                     return_dict=True).logits  # [chunk_size, 1]

            if logits_sum is None:
                logits_sum = chunk_logits.sum(dim=0)
            else:
                logits_sum = logits_sum + chunk_logits.sum(dim=0)

            # Free up GPU memory immediately
            del chunk_input_ids, chunk_attention_mask, chunk_logits
            torch.cuda.empty_cache()
            gc.collect()

        patient_logit = logits_sum / total_cells  # [1]
        return patient_logit.view(-1)

# class DataCollatorForPatientClassification(DataCollatorForCellClassification):
#     def __init__(self, token_dictionary, sample_id):
#         self.token_dictionary = token_dictionary
#         self.sample_id = sample_id

#     def __call__(self, batch):
#         # Group by patient
#         patient_to_cells = defaultdict(list)
#         patient_labels = {}

#         for item in batch:
#             logger.info(item)
#             patient_id = item[self.sample_id]
#             tokens = item['input_ids']
#             attention_mask = item['attention_mask']
#             label = item['label']

#             patient_to_cells[patient_id].append((tokens, attention_mask))
#             patient_labels[patient_id] = label

#         input_ids_batch = []
#         attention_mask_batch = []
#         labels = []

#         for patient_id, cell_data in patient_to_cells.items():
#             input_ids, attention_masks = zip(*cell_data)
#             input_ids = torch.stack(input_ids)
#             attention_masks = torch.stack(attention_masks)

#             input_ids_batch.append(input_ids)
#             attention_mask_batch.append(attention_masks)
#             labels.append(patient_labels[patient_id])

#         return {
#             'input_ids': input_ids_batch,  # list of [n_cells, seq_len]
#             'attention_mask': attention_mask_batch,
#             'labels': torch.tensor(labels)
#         }

# class BertMILClassifier(nn.Module):
#     def __init__(self, model_name_or_path, num_labels):
#         super().__init__()
#         self.bert = BertForSequenceClassification.from_pretrained(
#             model_name_or_path,
#             num_labels=num_labels,
#             output_attentions=False,
#             output_hidden_states=True  # We use hidden states for MIL
#         )
#         self.num_labels = num_labels

#     def forward(self, input_ids, attention_mask, labels=None):
#         pooled_outputs = []

#         for patient_input_ids, patient_mask in zip(input_ids, attention_mask):
#             outputs = self.bert.bert(
#                 input_ids=patient_input_ids.to(self.bert.device),
#                 attention_mask=patient_mask.to(self.bert.device)
#             )
#             cls_embeddings = outputs.last_hidden_state[:, 0, :]
#             pooled_embedding = cls_embeddings.mean(dim=0, keepdim=True)
#             pooled_outputs.append(pooled_embedding)

#         pooled_outputs = torch.cat(pooled_outputs, dim=0)
#         logits = self.bert.classifier(pooled_outputs)

#         loss = None
#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(logits, labels.to(logits.device))

#         return {'logits': logits, 'loss': loss}

def get_splits_cv(data_loader):
    """Get cross-validation splits"""
    cv = data_loader.cv_split_dict
    n_splits = cv['n_splits']
    id_column = cv['id_column']

    train_ids_list = []
    test_ids_list = []
    for i in range(n_splits):
        train_ids = cv[f'fold_{i+1}']['train_ids']
        test_ids = cv[f'fold_{i+1}']['test_ids']
        train_ids_list.append(train_ids)
        test_ids_list.append(test_ids)
    return id_column, n_splits, train_ids_list, test_ids_list

def save_results(pred_df, metrics_df, cls_report, saving_dir, postfix, viz=False, model_name=None, label_names=None):
    """Utility function to save results and visualizations"""
    import os
    if pred_df is not None:
        fname = join(saving_dir, f'cls_predictions_{postfix}.csv')
        pred_df.to_csv(fname)
    if metrics_df is not None:
        fname = join(saving_dir, f'cls_metrics_{postfix}.csv')
        metrics_df.to_csv(fname)
    if cls_report is not None:
        fname = join(saving_dir, f'cls_report_{postfix}.csv')
        pd.DataFrame(cls_report).transpose().to_csv(fname)
    if viz and pred_df is not None:
        fig = plot_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'], estimator_name=model_name, label_names=label_names)
        fname = join(saving_dir, f'cls_metrics_{postfix}.png')
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()

# def train_patient_classifier(train_ds, test_ds, sample_id, model_dir, vocab_dir, output_dir, num_labels, device):
#     # import logging
#     # logger = logging.getLogger(__name__)
#     logger.info('Training Patient-Level MIL Classifier')
#     with open(vocab_dir, "rb") as f:
#         vocab = pickle.load(f)
#     model = BertMILClassifier(model_dir, num_labels).to(device)
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         logging_dir=output_dir,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         num_train_epochs=3,
#         evaluation_strategy='epoch',
#         save_strategy='epoch'
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=DataCollatorForPatientClassification(token_dictionary=vocab, sample_id=sample_id),
#         train_dataset=train_ds,
#         eval_dataset=test_ds,
#         compute_metrics=None  # You can pass compute_metrics if needed
#     )
#     trainer.train()
#     trainer.save_model(output_dir)
#     predictions = trainer.predict(test_ds)
#     y_pred = predictions.predictions.argmax(axis=-1)
#     y_score = softmax(predictions.predictions, axis=-1)
#     y_true = np.array(test_ds['label'])
#     return trainer, y_true, y_pred, y_score

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Run model evaluation and return key prediction outputs.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Dataloader for evaluation set.
        device (torch.device): Torch device.
        threshold (float): Threshold for converting probabilities to binary labels.

    Returns:
        trainer (dict): Dictionary with metrics ('auc', 'acc').
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Binary predicted labels.
        y_score (np.ndarray): Raw sigmoid probabilities.
    """
    model.eval()
    y_true, y_score, sample_ids = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting batch {i}")
            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            bag_sizes = batch["bag_sizes"]
            labels = batch["labels"].to(device)
            sample_id = batch["sample_ids"][0]
            
            outputs = model(input_ids, masks, bag_sizes)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_score.extend(probs)
            y_true.extend(labels.cpu().numpy())
            sample_ids.append(sample_id)

    y_score = np.array(y_score)
    y_true = np.array(y_true)
    y_pred = (y_score > threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)

    metrics = {
        "auc": auc,
        "acc": acc,
    }

    logger.info(sample_ids)
    return metrics,sample_ids, y_true, y_pred, y_score

def train_model(model, dataloader, optimizer, device, epochs):
    """
    Train a PyTorch model with binary classification.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): The device to use (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        logger (logging.Logger): Logger for output.

    Returns:
        model (torch.nn.Module): The trained model.
    """
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for i, batch in enumerate(dataloader):
            print(f"--------------batch {i} --------------")
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            bag_sizes = batch["bag_sizes"]
            labels = batch["labels"].to(device)

            outputs = model(input_ids, masks, bag_sizes)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            print(f"done {i}")

        train_auc = roc_auc_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, [p > 0.5 for p in all_preds])

        logger.info(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | AUC: {train_auc:.4f} | ACC: {train_acc:.4f}")

    return model

def train_patient_classifier(train_ds, test_ds, sample_id, model_dir, vocab_dir, output_dir, num_labels, device, epochs):
    # import logging
    # logger = logging.getLogger(__name__)
    logger.info('Training Patient-Level MIL Classifier')
    
    train_patient_dataset = SubsampledPatientDataset(train_ds)
    val_patient_dataset = SubsampledPatientDataset(test_ds)
    dataloader = DataLoader(train_patient_dataset, batch_size=1, collate_fn=collate_patient_level)

    model = ChunkedBertMILClassifier_check(model_dir, device= device, num_labels=num_labels)
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # epochs=1
    
    model = train_model(model, dataloader, optimizer, device, epochs=epochs)
    
    test_dataloader = DataLoader(val_patient_dataset, batch_size=1, collate_fn=collate_patient_level)
    metrics, sample_ids, y_true, y_pred, y_score = evaluate_model(model, test_dataloader, device, threshold=0.5)
    logger.info(sample_ids)
    return model, sample_ids, y_true, y_pred, y_score

def train_classifier_cell(train_ds, test_ds, model_dir, vocab_dir, output_dir, num_labels, device, freeze_layers, batch_size, num_train_epochs):
    # logger = logging.getLogger(__name__)
    logger.info('Fine Tuning Model')
    training_args ={}
    training_args['output_dir'] = output_dir
    training_args['logging_dir'] = output_dir
    training_args['per_device_train_batch_size']=batch_size
    training_args['per_device_eval_batch_size']=batch_size
    training_args['num_train_epochs'] = num_train_epochs
    
    y_test = np.array(test_ds['label'])
    with open(vocab_dir, "rb") as f:
        vocab = pickle.load(f)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
        torch_dtype=torch.float32
    ).to(device)
    if freeze_layers > 0:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
    training_args_init = TrainingArguments(**training_args)
    trainer = Trainer(
        model= model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(token_dictionary=vocab),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=None  # You can pass compute_metrics if needed
    )
    trainer.train()
    trainer.save_model(output_dir)
    output = trainer.predict(test_ds)
    y_pred = output.predictions
    scores = output.predictions
    y_pred = scores.argmax(axis=-1)
    y_pred_score = softmax(scores, axis=-1)
    return trainer, y_test, y_pred, y_pred_score
