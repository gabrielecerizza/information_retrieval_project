import json
import os
from cv2 import LSD_REFINE_NONE
import torch
import traceback
"""
from irproject.historical_events import (
    get_bart_sentences_not_train, get_bart_sentences_train, 
    load_rams_data 
)
"""
from .historical_events import (
    get_bart_sentences_not_train, get_bart_sentences_train,
    get_bart_sentences_no_span_not_train, get_bart_sentences_no_span_train, 
    load_rams_data, get_event_names_dict
)
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from torch import nn
from torch.nn import (
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Dropout,
    Embedding, Linear, LSTM
)
from torch.nn import functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Accuracy, F1, Precision, Recall
from tqdm.notebook import tqdm
from transformers import (
    BartConfig, BartModel, BartTokenizer, BertModel, BertTokenizer
)
from transformers.file_utils import ModelOutput
from transformers.generation_utils import top_k_top_p_filtering
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from typing import Iterable, Optional

class MultiTaskLoss(nn.Module):
    """Uncertainty weighted loss for multi-task learning
    proposed in Kendall et al., Multi-Task Learning Using 
    Uncertainty to Weigh Losses for Scene Geometry and 
    Semantics, arXiv:1705.07115v3.

    Possible alternative implementations are available
    at:
        - https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
        - https://github.com/lorenmt/mtan/blob/master/im2im_pred/utils.py
    """

    def __init__(self, losses_num: int = 2, num_tokens_labels: int = 5):
        super(MultiTaskLoss, self).__init__()
        self.losses_num = losses_num
        self.num_tokens_labels = num_tokens_labels
        self.log_vars = nn.Parameter(torch.zeros((losses_num)))

    def forward(self, seq_clf_out, tokens_clf_out, labels, tokens_labels, attention_mask):

        loss_ce = CrossEntropyLoss()
        loss_bce = BCELoss()

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = tokens_clf_out.view(-1, self.num_tokens_labels)
            active_labels = torch.where(
                active_loss, 
                tokens_labels.view(-1), 
                torch.tensor(loss_ce.ignore_index).type_as(tokens_labels)
            )
            loss0 = loss_ce(active_logits, active_labels)
        else:
            loss0 = loss_ce(
                tokens_clf_out.view(-1, self.num_tokens_labels), 
                tokens_labels.view(-1)
            )

        loss1 = loss_bce(seq_clf_out.view(-1), labels.view(-1))

        losses = [loss0, loss1]

        loss = sum(
            torch.exp(-self.log_vars[i]) * losses[i] + (self.log_vars[i] / 2)
            for i in range(self.losses_num)
        )
        
        return loss

    
class MultiTaskLearningModel(LightningModule):
    def __init__(
        self, base_model = None, dropout_rate: float = 0.1, 
        hidden_size: int = 768, 
        num_tokens_labels: int = 5,
        average: str = "weighted"
    ):
        super(MultiTaskLearningModel, self).__init__()
        if base_model is None:
            self.base_model = BertModel.from_pretrained(
                "bert-base-cased"
            )
        else:
            self.base_model = base_model

        self.num_tokens_labels = num_tokens_labels

        # We could avoid sigmoid here and use the
        # BCEWithLogitsLoss, which computes both the sigmoid
        # and the BCE with a trick for numerical stability.
        self.seq_clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=768, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Sigmoid()
        )

        self.tokens_clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_tokens_labels)
            # nn.LogSoftmax(dim=1)
        )

        self.average = average

        self.multi_loss = MultiTaskLoss(2, num_tokens_labels)
        
        # For binary classification we need to set
        # num_classes to 1. See:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5705
        self.seqc_accuracy = Accuracy(
            num_classes=1,
            average=self.average
        )
        self.tokc_accuracy = Accuracy(
            num_classes=num_tokens_labels,
            ignore_index=(num_tokens_labels - 1), # ignore non-entities
            average=self.average
        )
        self.seqc_f1 = F1(
            num_classes=1,
            average=self.average
        )
        self.tokc_f1 = F1( 
            num_classes=num_tokens_labels, 
            ignore_index=(num_tokens_labels - 1), # ignore non-entities
            average=self.average
        )
        self.seqc_prec = Precision(
            num_classes=1,
            average=self.average
        )
        self.seqc_recall = Recall(
            num_classes=1,
            average=self.average
        )

        for param in self.seq_clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for param in self.tokens_clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = output.last_hidden_state

        # TODO Since we are using mean, maybe it's better not 
        # to pad to max?
        seq_clf_out = self.seq_clf(
            torch.mean(last_hidden_state, dim=1)
        )
        tokens_clf_out = self.tokens_clf(last_hidden_state)

        return seq_clf_out, tokens_clf_out

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)
        loss = self.multi_loss(seqc_out, tokc_out, labels, tokens_labels, attention_mask)

        # Some metrics want only positive values. We
        # replace -100 with the last token label and tell
        # the metric to ignore score for this label.
        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1, 
            tokens_labels
        )

        tokc_preds = torch.argmax(tokc_out, -1)

        seqc_acc = self.seqc_accuracy(seqc_out.view(-1), labels.int().view(-1))
        tokc_acc = self.tokc_accuracy(tokc_preds, tokens_labels)

        try:
            seqc_f1 = self.seqc_f1(seqc_out, labels.int())
        except Exception as ex:
            traceback.print_exc()
            print("Error in train seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in train tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)
        loss = self.multi_loss(seqc_out, tokc_out, labels, tokens_labels, attention_mask)

        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1, 
            tokens_labels
        )

        tokc_preds = torch.argmax(tokc_out, -1)

        seqc_acc = self.seqc_accuracy(seqc_out.view(-1), labels.int().view(-1))
        tokc_acc = self.tokc_accuracy(tokc_preds, tokens_labels)

        try:
            seqc_f1 = self.seqc_f1(seqc_out, labels.int())
        except Exception as ex:
            traceback.print_exc()
            print("Error in val seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in val tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)

        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1,
            tokens_labels
        )

        tokc_preds = torch.argmax(tokc_out, -1)

        seqc_acc = self.seqc_accuracy(
            seqc_out.view(-1), labels.int().view(-1)
        )
        tokc_acc = self.tokc_accuracy(
            tokc_preds, tokens_labels
        )

        try:
            seqc_f1 = self.seqc_f1(seqc_out, labels.int())
        except Exception as ex:
            traceback.print_exc()
            print("Error in test seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in test tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)

        self.log("test_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)

        seqc_prec = self.seqc_prec(seqc_out, labels.int())
        seqc_recall = self.seqc_recall(seqc_out, labels.int())

        return seqc_acc, tokc_acc, seqc_f1, tokc_f1, seqc_prec, seqc_recall

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/document_classification"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_seqc_acc": self.seqc_accuracy.compute().item(),
            "final_tokc_acc": self.tokc_accuracy.compute().item(),
            "final_seqc_f1": self.seqc_f1.compute().item(),
            "final_tokc_f1": self.tokc_f1.compute().item(),
            "final_seqc_prec": self.seqc_prec.compute().item(),
            "final_seqc_recall": self.seqc_recall.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                    "seqc_acc": tup[0].item(), 
                    "tokc_acc": tup[1].item(), 
                    "seqc_f1": tup[2].item(), 
                    "tokc_f1": tup[3].item(),
                    "seqc_prec": tup[4].item(),
                    "seqc_recall": tup[5].item()
                }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/mtl_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)

    def freeze_base(self):
        for param in self.base_model.named_parameters():
            param[1].requires_grad=False
        print("Base frozen.")

    def unfreeze_base(self):
        for param in self.base_model.named_parameters():
            param[1].requires_grad=True
        print("Base unfrozen.")


class WikiDataModule(LightningDataModule):

    def __init__(
        self, data_dir: str = "datasets/historical_events",
        batch_size: int = 128, num_workers: int = 0, 
        num_tokens_labels: int = 5
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = 512
        self.num_tokens_labels = num_tokens_labels

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, data_split: str):
        return {
            "input_ids": torch.load(f"{self.data_dir}/{data_split}/input_ids.pkl"),
            "attention_mask": torch.load(f"{self.data_dir}/{data_split}/attention_mask.pkl"),
            "tokens_labels": torch.load(f"{self.data_dir}/{data_split}/tokens_labels.pkl"),
            "labels": torch.load(f"{self.data_dir}/{data_split}/labels.pkl"),
            "tag2idx": torch.load(f"{self.data_dir}/{data_split}/tag2idx.pkl"),
            "idx2tag": torch.load(f"{self.data_dir}/{data_split}/idx2tag.pkl")   
        }

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("valid")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        wiki_train = TensorDataset(
            torch.tensor(self.train_dict["input_ids"]), 
            torch.tensor(self.train_dict["attention_mask"]),
            torch.tensor(self.train_dict["tokens_labels"]),
            torch.tensor(self.train_dict["labels"], dtype=torch.float32)
        )

        return DataLoader(
            wiki_train, batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        wiki_valid = TensorDataset(
            torch.tensor(self.valid_dict["input_ids"]), 
            torch.tensor(self.valid_dict["attention_mask"]),
            torch.tensor(self.valid_dict["tokens_labels"]),
            torch.tensor(self.valid_dict["labels"], dtype=torch.float32)
        )

        return DataLoader(
            wiki_valid, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        wiki_test = TensorDataset(
            torch.tensor(self.test_dict["input_ids"]), 
            torch.tensor(self.test_dict["attention_mask"]),
            torch.tensor(self.test_dict["tokens_labels"]),
            torch.tensor(self.test_dict["labels"], dtype=torch.float32)
        )
        

        return DataLoader(
            wiki_test, batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class FreezeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch <= 3:
            print(f"Epoch number {epoch}, freezing base.")
            pl_module.freeze_base()
        else:
            print(f"Epoch number {epoch}, unfreezing base.")
            pl_module.unfreeze_base()


class RAMSDataset(Dataset):
    def __init__(
        self, input_ids, attention_masks, spans, 
        spans_trg_true, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys, doc_tokens,
        span_mappers
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.spans = spans
        self.spans_trg_true = spans_trg_true
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys
        self.doc_tokens = doc_tokens
        self.span_mappers = span_mappers

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], \
            self.spans[idx], self.spans_trg_true[idx], \
            self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx], \
            self.doc_tokens[idx], self.span_mappers[idx]


class RAMSArgumentDataset(Dataset):
    def __init__(
        self, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys
    ):
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        return self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx]


class RAMSEventGenDataset(Dataset):
    def __init__(
        self, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys
    ):
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        return self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx]


def collate_RAMS(batch):
    tokens_ids = torch.tensor(
        [item[0] for item in batch]
    )
    attention_masks = torch.tensor(
        [item[1] for item in batch]
    )
    spans = [item[2] for item in batch]
    spans_trg_labels = [item[3] for item in batch]
    encoder_input_ids = torch.tensor(
        [item[4] for item in batch]
    )
    encoder_attention_mask = torch.tensor(
        [item[5] for item in batch] 
    )
    dec_input_ids = torch.tensor(
        [item[6] for item in batch]
    )
    dec_attention_mask = torch.tensor(
        [item[7] for item in batch]
    )
    doc_keys = [item[8] for item in batch]
    doc_tokens = [item[9] for item in batch]
    span_mappers = [item[10] for item in batch]

    return {
        "input_ids": tokens_ids, 
        "attention_masks": attention_masks,
        "spans": spans, 
        "spans_trg_true": list2tensor(spans_trg_labels),
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "doc_keys": doc_keys,
        "doc_tokens": doc_tokens,
        "span_mappers": span_mappers
    }


def collate_argument_RAMS(batch):
    encoder_input_ids = torch.tensor(
        [item[0] for item in batch]
    )
    encoder_attention_mask = torch.tensor(
        [item[1] for item in batch] 
    )
    dec_input_ids = torch.tensor(
        [item[2] for item in batch]
    )
    dec_attention_mask = torch.tensor(
        [item[3] for item in batch]
    )
    doc_keys = [item[4] for item in batch]

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "doc_keys": doc_keys
    }


class RAMSDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/rams",
        batch_size: int = 16, num_workers: int = 0,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.rams_train = RAMSDataset(
                input_ids=self.train_dict["tokens_ids"], 
                attention_masks=self.train_dict["attention_masks"], 
                spans=self.train_dict["spans"], 
                spans_trg_true=self.train_dict["spans_trg_labels"],
                encoder_input_ids=self.train_dict["args_ids"], 
                encoder_attention_mask=self.train_dict["args_masks"],
                dec_input_ids=self.train_dict["args_dec_ids"],
                dec_attention_mask=self.train_dict["args_dec_masks"],
                doc_keys=self.train_dict["doc_keys"],
                doc_tokens=self.train_dict["tokens"],
                span_mappers=self.train_dict["span_mappers"]
            )

            self.rams_valid = RAMSDataset(
                input_ids=self.valid_dict["tokens_ids"], 
                attention_masks=self.valid_dict["attention_masks"], 
                spans=self.valid_dict["spans"], 
                spans_trg_true=self.valid_dict["spans_trg_labels"],
                encoder_input_ids=self.valid_dict["args_ids"], 
                encoder_attention_mask=self.valid_dict["args_masks"],
                dec_input_ids=self.valid_dict["args_dec_ids"],
                dec_attention_mask=self.valid_dict["args_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"],
                doc_tokens=self.valid_dict["tokens"],
                span_mappers=self.valid_dict["span_mappers"]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.rams_test = RAMSDataset(
                input_ids=self.test_dict["tokens_ids"], 
                attention_masks=self.test_dict["attention_masks"], 
                spans=self.test_dict["spans"], 
                spans_trg_true=self.test_dict["spans_trg_labels"],
                encoder_input_ids=self.test_dict["args_ids"], 
                encoder_attention_mask=self.test_dict["args_masks"],
                dec_input_ids=self.test_dict["args_dec_ids"],
                dec_attention_mask=self.test_dict["args_dec_masks"],
                doc_keys=self.test_dict["doc_keys"],
                doc_tokens=self.test_dict["tokens"],
                span_mappers=self.test_dict["span_mappers"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_RAMS
        ) 


class RAMSArgumentDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/rams",
        batch_size: int = 16, num_workers: int = 0,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.rams_train = RAMSArgumentDataset(
                encoder_input_ids=self.train_dict["args_ids"], 
                encoder_attention_mask=self.train_dict["args_masks"],
                dec_input_ids=self.train_dict["args_dec_ids"],
                dec_attention_mask=self.train_dict["args_dec_masks"],
                doc_keys=self.train_dict["doc_keys"]
            )

            self.rams_valid = RAMSArgumentDataset(
                encoder_input_ids=self.valid_dict["args_ids"], 
                encoder_attention_mask=self.valid_dict["args_masks"],
                dec_input_ids=self.valid_dict["args_dec_ids"],
                dec_attention_mask=self.valid_dict["args_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.rams_test = RAMSArgumentDataset(
                encoder_input_ids=self.test_dict["args_ids"], 
                encoder_attention_mask=self.test_dict["args_masks"],
                dec_input_ids=self.test_dict["args_dec_ids"],
                dec_attention_mask=self.test_dict["args_dec_masks"],
                doc_keys=self.test_dict["doc_keys"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_argument_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        ) 


class RAMSEventGenDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/rams",
        batch_size: int = 16, num_workers: int = 0,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.rams_train = RAMSEventGenDataset(
                encoder_input_ids=self.train_dict["evt_ids"], 
                encoder_attention_mask=self.train_dict["evt_masks"],
                dec_input_ids=self.train_dict["evt_dec_ids"],
                dec_attention_mask=self.train_dict["evt_dec_masks"],
                doc_keys=self.train_dict["doc_keys"]
            )

            self.rams_valid = RAMSEventGenDataset(
                encoder_input_ids=self.valid_dict["evt_ids"], 
                encoder_attention_mask=self.valid_dict["evt_masks"],
                dec_input_ids=self.valid_dict["evt_dec_ids"],
                dec_attention_mask=self.valid_dict["evt_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.rams_test = RAMSEventGenDataset(
                encoder_input_ids=self.test_dict["evt_ids"], 
                encoder_attention_mask=self.test_dict["evt_masks"],
                dec_input_ids=self.test_dict["evt_dec_ids"],
                dec_attention_mask=self.test_dict["evt_dec_masks"],
                doc_keys=self.test_dict["doc_keys"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_argument_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        ) 


def list2tensor(lss):
    res = torch.tensor([])
    for ls in lss:
        res = torch.cat([res, torch.tensor(ls)])
    return res


class EventModel(LightningModule):
    def __init__(
        self,
        bert,
        tokenizer,
        bart_tokenizer,
        data_dir: str = "datasets/rams",
        ontology_dir: str = "datasets",
        num_events: int = 140, # 139 events + no_evt 
        dropout_rate: float = 0.2,
        bert_hidden_dims: int = 768, 
        hidden_dims: int = 150,
        max_span_length: int = 3,
        width_embedding_dim: int = 768,
        average: str = "weighted"
    ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.bart_tokenizer.add_tokens([" <arg>", " <trg>"])
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir

        # self.width_embedding = nn.Embedding(
        #    max_span_length+1, width_embedding_dim
        # )

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(bert_hidden_dims * 3, hidden_dims),
            # nn.Linear(bert_hidden_dims * 3 + width_embedding_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_events)
        )

        self.idx2evt, self.evt2idx = self.load_event_dicts()

        self.average = average

        self.accuracy = Accuracy(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.f1 = F1(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.prec = Precision(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.recall = Recall(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )

        self.accuracy_full = Accuracy(
            num_classes=num_events,
            average=average
        )
        self.f1_full = F1(
            num_classes=num_events,
            average=average
        )
        self.prec_full = Precision(
            num_classes=num_events,
            average=average
        )
        self.recall_full = Recall(
            num_classes=num_events,
            average=average
        )

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def load_event_dicts(self):
        _, data_dicts = load_rams_data(self.data_dir)

        return data_dicts["idx2evt"], data_dicts["evt2idx"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-5
        )
        return optimizer

    def score(self, span_logits, span_true):
        span_preds = torch.argmax(span_logits, -1)
        acc = self.accuracy(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1 = self.f1(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec = self.prec(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall = self.recall(
            span_preds.view(-1), span_true.view(-1).int()
        )

        acc_f = self.accuracy_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1_f = self.f1_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec_f = self.prec_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall_f = self.recall_full(
            span_preds.view(-1), span_true.view(-1).int()
        )

        scores = {
            "acc": acc,
            "f1": f1,
            "prec": prec, 
            "recall": recall, 
            "acc_f": acc_f, 
            "f1_f": f1_f, 
            "prec_f": prec_f, 
            "recall_f": recall_f
        }

        return scores

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        span_batch,
        span_mappers,
        doc_tokens,
        training: bool = False
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = output.last_hidden_state
        span_embeddings = []
        span_lists_lengths = []

        for batch_num, span_list in enumerate(span_batch):
            span_lists_lengths.append(len(span_list))

            batch_hidden_state = last_hidden_state[batch_num]
            batch_span_embs = []

            for span in span_list:
                span_start, span_end, span_width = span
                first_word_emb = batch_hidden_state[span_start]
                last_word_emb = batch_hidden_state[span_end]

                span_width = torch.tensor([span_width], device=self.device)
                # span_width_embedding = self.width_embedding(span_width)

                mean_span = torch.mean(
                    batch_hidden_state[
                        span_start:span_end + 1
                    ],
                    dim=0
                )

                span_emb = torch.cat(
                    [
                        first_word_emb,
                        last_word_emb,
                        mean_span,
                        # span_width_embedding.view(-1)
                    ],
                    dim = -1
                )
                batch_span_embs.append(span_emb)

            span_embeddings.append(
                torch.stack(batch_span_embs)
            )

        span_embeddings = torch.cat(span_embeddings)
        logits = self.clf(span_embeddings)

        # Get the event name for each batch.
        evt_names = []
        spans_with_evts = []
        start = 0
        for spans_len in span_lists_lengths:
            end = start + spans_len
            span_logits = logits[start:end + 1]
            # Probs are not actually probabilities, since
            # we did not apply softmax, but for our
            # purposes they fulfill the same role.
            probs, evt_ids = torch.max(span_logits, -1)
            range_ids = torch.tensor(list(range(len(probs))))

            assert len(range_ids) == len(probs), \
                (len(range_ids), len(probs))
            
            # We check only the events that are not non-events.
            probs = probs[evt_ids > 0]
            range_ids = range_ids[evt_ids > 0]
            evt_ids = evt_ids[evt_ids > 0]

            if len(probs) == 0:
                # No events found. This is wrong in training,
                # since every instance has one event. We
                # pick a random event for a random span.
                # TODO Make it so that when we are not training, if
                # there are no events, we stop here. We cannot
                # produce a template for the argument model
                # anyway.
                evt_names.append(
                    self.idx2evt[1 + torch.randint(139, (1,)).item()] # + 1 to avoid no_evt
                )
                spans_with_evts.append(
                    torch.randint(spans_len - 1, (1,)).item()
                )
            else:
                max_prob, max_prob_id, max_range_id = -1000, -1, -1
                for prob, prob_id, range_id in zip(
                    probs, evt_ids, range_ids
                ):
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_id = prob_id
                        max_range_id = range_id
                
                evt_names.append(
                    # (max_prob, self.idx2evt[max_prob_id.item()])
                    self.idx2evt[max_prob_id.item()]
                )
                if max_range_id == spans_len:
                    # print("probs", probs)
                    # print("evt_ids", evt_ids)
                    # print("range_ids", range_ids)
                    max_range_id = max_range_id - 1

                spans_with_evts.append(max_range_id)
            start = end

        assert len(evt_names) == len(spans_with_evts), \
            (len(evt_names), len(spans_with_evts))

        enc_ids = None
        enc_attn_masks = None
        enc_sentences = None

        # print("evt_names", evt_names)

        if training:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_train(
                input_ids=input_ids,
                span_batch=span_batch,
                spans_with_evts=spans_with_evts,
                evt_names=evt_names,
                doc_tokens=doc_tokens,
                span_mappers=span_mappers,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )
        else:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_not_train(
                input_ids=input_ids,
                span_batch=span_batch,
                spans_with_evts=spans_with_evts,
                evt_names=evt_names,
                bert_tokenizer=self.tokenizer,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )

        return logits, evt_names, enc_ids, enc_attn_masks

    def training_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, spans_trg_true
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", scores["acc"], prog_bar=True)
        self.log("train_f1", scores["f1"], prog_bar=True)
        self.log("train_prec", scores["prec"], prog_bar=True)
        self.log("train_recall", scores["recall"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, spans_trg_true
        )

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", scores["acc"], prog_bar=True)
        self.log("valid_f1", scores["f1"], prog_bar=True)
        self.log("valid_prec", scores["prec"], prog_bar=True)
        self.log("valid_recall", scores["recall"], prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=False
        )

        spans_trg_true = batch["spans_trg_true"]

        scores = self.score(
            logits, spans_trg_true
        )
        return list(scores.values())

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/events"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "final_acc_f": self.accuracy_full.compute().item(),
            "final_f1_f": self.f1_full.compute().item(),
            "final_prec_f": self.prec_full.compute().item(),
            "final_recall_f": self.recall_full.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item(),
                "acc_f": tup[4].item(),  
                "f1_f": tup[5].item(),
                "prec_f": tup[6].item(),
                "recall_f": tup[7].item()
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/event_extraction_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


class EventBiLSTMModel(LightningModule):
    def __init__(
        self,
        bert,
        tokenizer,
        bart_tokenizer,
        data_dir: str = "datasets/rams",
        ontology_dir: str = "datasets",
        num_events: int = 140, # 139 events + no_evt 
        dropout_rate: float = 0.2,
        bert_hidden_dims: int = 768, 
        hidden_dims: int = 150,
        max_span_length: int = 3,
        lstm_dim: int = 768,
        num_layers: int = 3,
        freeze_bert: bool = False,
        average: str = "weighted"
    ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.bart_tokenizer.add_tokens([" <arg>", " <trg>"])
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir

        # self.width_embedding = nn.Embedding(
        #    max_span_length+1, width_embedding_dim
        # )
        if freeze_bert:
            for param in self.bert.named_parameters():
                param[1].requires_grad=False

        self.lstm = LSTM(
            input_size=768,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, hidden_dims),
            # nn.Linear(bert_hidden_dims * 3 + width_embedding_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dims, num_events)
        )

        self.idx2evt, self.evt2idx = self.load_event_dicts()

        self.average = average

        self.accuracy = Accuracy(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.f1 = F1(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.prec = Precision(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.recall = Recall(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )

        self.accuracy_full = Accuracy(
            num_classes=num_events,
            average=average
        )
        self.f1_full = F1(
            num_classes=num_events,
            average=average
        )
        self.prec_full = Precision(
            num_classes=num_events,
            average=average
        )
        self.recall_full = Recall(
            num_classes=num_events,
            average=average
        )

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def load_event_dicts(self):
        _, data_dicts = load_rams_data(self.data_dir)

        return data_dicts["idx2evt"], data_dicts["evt2idx"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-5
        )
        return optimizer

    def score(self, span_logits, span_true):
        span_preds = torch.argmax(span_logits, -1)
        acc = self.accuracy(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1 = self.f1(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec = self.prec(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall = self.recall(
            span_preds.view(-1), span_true.view(-1).int()
        )

        acc_f = self.accuracy_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1_f = self.f1_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec_f = self.prec_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall_f = self.recall_full(
            span_preds.view(-1), span_true.view(-1).int()
        )

        scores = {
            "acc": acc,
            "f1": f1,
            "prec": prec, 
            "recall": recall, 
            "acc_f": acc_f, 
            "f1_f": f1_f, 
            "prec_f": prec_f, 
            "recall_f": recall_f
        }

        return scores

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        span_batch,
        span_mappers,
        doc_tokens,
        training: bool = False
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = output.last_hidden_state
    
        out, _ = self.lstm(
            last_hidden_state
        )
        out = torch.mean(out, -1)

        logits = self.clf(out)

        evt_ids = torch.argmax(logits, -1)
        evt_names = []
        for evt_id in evt_ids:
            evt_names.append(self.idx2evt[evt_id.item()])
        
        enc_ids = None
        enc_attn_masks = None
        enc_sentences = None

        if training:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_no_span_train(
                input_ids=input_ids,
                evt_names=evt_names,
                doc_tokens=doc_tokens,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )
        else:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_no_span_not_train(
                input_ids=input_ids,
                evt_names=evt_names,
                bert_tokenizer=self.tokenizer,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )

        return logits, evt_names, enc_ids, enc_attn_masks

    def training_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, docs_trg_true
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", scores["acc"], prog_bar=True)
        self.log("train_f1", scores["f1"], prog_bar=True)
        self.log("train_prec", scores["prec"], prog_bar=True)
        self.log("train_recall", scores["recall"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, docs_trg_true
        )

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", scores["acc"], prog_bar=True)
        self.log("valid_f1", scores["f1"], prog_bar=True)
        self.log("valid_prec", scores["prec"], prog_bar=True)
        self.log("valid_recall", scores["recall"], prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=False
        )

        spans_trg_true = batch["spans_trg_true"]
        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]

        scores = self.score(
            logits, docs_trg_true
        )
        return list(scores.values())

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/events"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "final_acc_f": self.accuracy_full.compute().item(),
            "final_f1_f": self.f1_full.compute().item(),
            "final_prec_f": self.prec_full.compute().item(),
            "final_recall_f": self.recall_full.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item(),
                "acc_f": tup[4].item(),  
                "f1_f": tup[5].item(),
                "prec_f": tup[6].item(),
                "recall_f": tup[7].item()
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/event_bilstm_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


class EventBertModel(LightningModule):
    def __init__(
        self,
        bert,
        tokenizer,
        bart_tokenizer,
        data_dir: str = "datasets/rams",
        ontology_dir: str = "datasets",
        num_events: int = 140, # 139 events + no_evt 
        dropout_rate: float = 0.4,
        bert_hidden_dims: int = 768, 
        hidden_dims: int = 150,
        max_span_length: int = 3,
        average: str = "weighted"
    ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.bart_tokenizer.add_tokens([" <arg>", " <trg>"])
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir

        # self.width_embedding = nn.Embedding(
        #    max_span_length+1, width_embedding_dim
        # )

        """
        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(768, hidden_dims),
            # nn.Linear(bert_hidden_dims * 3 + width_embedding_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_events)
        )
        """
        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(768, num_events)
        )

        self.idx2evt, self.evt2idx = self.load_event_dicts()

        self.average = average

        self.accuracy = Accuracy(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.f1 = F1(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.prec = Precision(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.recall = Recall(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )

        self.accuracy_full = Accuracy(
            num_classes=num_events,
            average=average
        )
        self.f1_full = F1(
            num_classes=num_events,
            average=average
        )
        self.prec_full = Precision(
            num_classes=num_events,
            average=average
        )
        self.recall_full = Recall(
            num_classes=num_events,
            average=average
        )

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def load_event_dicts(self):
        _, data_dicts = load_rams_data(self.data_dir)

        return data_dicts["idx2evt"], data_dicts["evt2idx"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-5
        )
        return optimizer

    def score(self, span_logits, span_true):
        span_preds = torch.argmax(span_logits, -1)
        acc = self.accuracy(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1 = self.f1(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec = self.prec(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall = self.recall(
            span_preds.view(-1), span_true.view(-1).int()
        )

        acc_f = self.accuracy_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        f1_f = self.f1_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        prec_f = self.prec_full(
            span_preds.view(-1), span_true.view(-1).int()
        )
        recall_f = self.recall_full(
            span_preds.view(-1), span_true.view(-1).int()
        )

        scores = {
            "acc": acc,
            "f1": f1,
            "prec": prec, 
            "recall": recall, 
            "acc_f": acc_f, 
            "f1_f": f1_f, 
            "prec_f": prec_f, 
            "recall_f": recall_f
        }

        return scores

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        span_batch,
        span_mappers,
        doc_tokens,
        training: bool = False
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        out = output[1]
        logits = self.clf(out)

        evt_ids = torch.argmax(logits, -1)
        evt_names = []
        for evt_id in evt_ids:
            if evt_id == 0:
                evt_names.append(
                    self.idx2evt[1 + torch.randint(139, (1,)).item()]
                )
            else:
                evt_names.append(self.idx2evt[evt_id.item()])
        
        enc_ids = None
        enc_attn_masks = None
        enc_sentences = None

        if training:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_no_span_train(
                input_ids=input_ids,
                evt_names=evt_names,
                doc_tokens=doc_tokens,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )
        else:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_no_span_not_train(
                input_ids=input_ids,
                evt_names=evt_names,
                bert_tokenizer=self.tokenizer,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )

        return logits, evt_names, enc_ids, enc_attn_masks

    def training_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, docs_trg_true
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", scores["acc"], prog_bar=True)
        self.log("train_f1", scores["f1"], prog_bar=True)
        self.log("train_prec", scores["prec"], prog_bar=True)
        self.log("train_recall", scores["recall"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )

        scores = self.score(
            logits, docs_trg_true
        )

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", scores["acc"], prog_bar=True)
        self.log("valid_f1", scores["f1"], prog_bar=True)
        self.log("valid_prec", scores["prec"], prog_bar=True)
        self.log("valid_recall", scores["recall"], prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=False
        )

        spans_trg_true = batch["spans_trg_true"]
        evt_ids = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_ids]

        scores = self.score(
            logits, docs_trg_true
        )
        return list(scores.values())

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/events"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "final_acc_f": self.accuracy_full.compute().item(),
            "final_f1_f": self.f1_full.compute().item(),
            "final_prec_f": self.prec_full.compute().item(),
            "final_recall_f": self.recall_full.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item(),
                "acc_f": tup[4].item(),  
                "f1_f": tup[5].item(),
                "prec_f": tup[6].item(),
                "recall_f": tup[7].item()
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/event_bert_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


class JointModel(LightningModule):
    """Not tested since the model crashed on training.
    """
    def __init__(self, event_model, argument_model):
        super(JointModel, self).__init__()
        self.event_model = event_model
        self.argument_model = argument_model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        evt_logits, evt_names, enc_ids, enc_attn_masks = self.event_model(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        loss_ce = CrossEntropyLoss()

        evt_loss = loss_ce(
            evt_logits.view(-1, evt_logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        arg_logits, arg_labels = self.argument_model(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        arg_loss = loss_ce(
            arg_logits.view(-1, arg_logits.shape[-1]), 
            arg_labels.view(-1)
        )

        arg_loss = torch.mean(arg_loss)

        evt_acc, evt_f1, evt_prec, evt_recall = self.event_model.score(
            evt_logits, spans_trg_true
        )

        self.log("train_evt_loss", evt_loss, prog_bar=True)
        self.log("train_arg_loss", arg_loss, prog_bar=True)
        self.log("train_evt_acc", evt_acc, prog_bar=True)
        self.log("train_evt_f1", evt_f1, prog_bar=True)
        self.log("train_evt_prec", evt_prec, prog_bar=True)
        self.log("train_evt_recall", evt_recall, prog_bar=True)

        loss = evt_loss + arg_loss
        return loss

    def validation_step(self, batch, batch_idx):
        evt_logits, evt_names, enc_ids, enc_attn_masks = self.event_model(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        loss_ce = CrossEntropyLoss()

        evt_loss = loss_ce(
            evt_logits.view(-1, evt_logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        arg_logits, arg_labels = self.argument_model(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        arg_loss = loss_ce(
            arg_logits.view(-1, arg_logits.shape[-1]), 
            arg_labels.view(-1)
        )

        arg_loss = torch.mean(arg_loss)

        evt_acc, evt_f1, evt_prec, evt_recall = self.event_model.score(
            evt_logits, spans_trg_true
        )

        self.log("valid_evt_loss", evt_loss, prog_bar=True)
        self.log("valid_arg_loss", arg_loss, prog_bar=True)
        self.log("valid_evt_acc", evt_acc, prog_bar=True)
        self.log("valid_evt_f1", evt_f1, prog_bar=True)
        self.log("valid_evt_prec", evt_prec, prog_bar=True)
        self.log("valid_evt_recall", evt_recall, prog_bar=True)

        loss = evt_loss + arg_loss
        return loss

    def test_step(self, batch, batch_idx):
        evt_logits, evt_names, enc_ids, enc_attn_masks = self.event_model(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        loss_ce = CrossEntropyLoss()

        evt_loss = loss_ce(
            evt_logits.view(-1, evt_logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        arg_logits, arg_labels = self.argument_model(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        arg_loss = loss_ce(
            arg_logits.view(-1, arg_logits.shape[-1]), 
            arg_labels.view(-1)
        )
        arg_loss = torch.mean(arg_loss)

        loss = evt_loss + arg_loss
        return loss


class ArgumentModelWrapper(LightningModule):
    def __init__(self, bart, bart_tokenizer):
        super(ArgumentModelWrapper, self).__init__()
        self.model = ArgumentModel(
            BartConfig.from_pretrained("facebook/bart-base"),
            bart=bart,
            bart_tokenizer=bart_tokenizer
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(
        self, 
        encoder_input_ids, 
        encoder_attention_mask,
        dec_input_ids,
        dec_attention_mask,
        training: bool = False
    ):
        return self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=dec_attention_mask,
            training=training
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            labels.view(-1)
        )

        loss = torch.mean(loss)
    
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            labels.view(-1)
        )

        loss = torch.mean(loss)
        self.log("valid_loss", loss, prog_bar=True)
    
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"]
        )

        filled_templates = self.model.generate(
            batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"], 
            do_sample=True, 
            top_k=20, 
            top_p=0.95, 
            max_length=30, 
            num_return_sequences=1,
            num_beams=1,
            repetition_penalty=1
        )

        return (
            batch["doc_keys"], 
            filled_templates, 
            batch["dec_input_ids"]
        )

    def test_epoch_end(self, outputs):
        os.makedirs(
            "results/historical_events/arguments",
            exist_ok=True
        )

        with open(
            "results/historical_events/arguments/predictions.jsonl",
            "w"
        ) as writer:
            for tup in outputs:
                pred = {
                    "doc_key": tup[0],
                    "predicted": self.model.tokenizer.decode(
                        tup[1].squeeze(0), 
                        skip_special_tokens=True
                    ),
                    "gold": self.model.tokenizer.decode(
                        tup[2].squeeze(0), 
                        skip_special_tokens=True
                    ) 
                }
                writer.write(json.dumps(pred)+'\n')

        return {} 


class ArgumentModel(PreTrainedModel):
    """Code adapted from the paper: 
    Li et al. - Document-Level Event Argument Extraction 
    by Conditional Generation, in Proceedings of the 2021 
    Conference of the North American Chapter of the 
    Association for Computational Linguistics: Human 
    Language Technologies (2021).

    https://github.com/raspberryice/gen-arg/
    
    """

    def __init__(self, config, bart, bart_tokenizer):
        super(ArgumentModel, self).__init__(config)
        self.config = config
        self.tokenizer = bart_tokenizer
        self.tokenizer.add_tokens([" <arg>"," <tgr>"])

        self.transformer = bart
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = self.config.vocab_size = len(self.tokenizer)

    def forward(
        self, 
        input_ids, 
        attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None, # from here, parameters passed by generate
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_embeds=None,
        training: bool = False # custom parameter to discriminate training ouputs and generate outputs
    ):
        # The decoder takes the sequence, starting from
        # the special start token, and predicts the next
        # token in the sequence. The labels are thus the
        # decoder input tokens shifted one left (in other
        # words, the decoder input tokens are shifted one
        # right wrt the labels).
        labels = None
        if training:
            labels = decoder_input_ids[:, 1:].clone() 
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_attention_mask = decoder_attention_mask[:, :-1]

            # Pad tokens must be manually replaced with -100:
            # https://discuss.huggingface.co/t/is-there-a-way-to-return-the-decoder-input-ids-from-tokenizer-prepare-seq2seq-batch/2929/3
            labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask= decoder_attention_mask,
            use_cache=use_cache, 
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        decoder_output = outputs.last_hidden_state  # (batch, dec_seq_len, hidden_dim)
        encoder_output = outputs.encoder_last_hidden_state  # (batch, enc_seq_len, hidden_dim)

        # We are taking only the embeddings of the input tokens 
        # to limit the vocabulary probabilities to those tokens.
        # In this way, we will only generate words that were passed
        # as input to the encoder.
        if input_embeds == None:  # input_embeds is passed by generate.  
            input_tokens_emb = self.transformer.encoder.embed_tokens(
                input_ids) * self.transformer.encoder.embed_scale  # (batch, enc_seq_len, hidden_dim)
        else:
            input_tokens_emb = input_embeds

        # Compute the formula from the paper:
        # h_i^T Emb(w)
        # We have to do this for the elements in the batch.
        # This is basically Batch Matrix Multiplication (BMM).
        # For each batch we have the probability of each of
        # the words in the encoder input (enc_seq_len) to
        # be generated by the decoder (dec_seq_len).   
        prod = torch.einsum(
            "bij,bjk->bik", 
            decoder_output, 
            torch.transpose(input_tokens_emb, 1, 2)
        )  # (batch, dec_seq_len, enc_seq_len)

        batch_size = prod.size(0)
        dec_seq_len = prod.size(1)
        logits = torch.full(
            (
                batch_size, 
                dec_seq_len, 
                self.transformer.config.vocab_size
            ), 
            fill_value=-1000,
            dtype=prod.dtype
        ).to(prod.device)

        # Possible duplicate indices.
        index = input_ids.unsqueeze(dim=1).expand_as(prod)

        # With dim=2 this is equivalent to:
        # self[i][j][index[i][j][k]] = src[i][j][k]
        logits.scatter_(dim=2, index=index, src=prod)

        if training:
            return logits, labels
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def get_encoder(self):
        return self.transformer.encoder

    def get_output_embeddings(self):
        # this method is needed for generation
        vocab_size, emb_size = self.transformer.shared.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = self.transformer.shared.weight.data
        return lin_layer 

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, input_embeds, encoder_input_ids, **kwargs):
        return {
            "input_ids": encoder_input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "input_embeds": input_embeds,
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    # # this is a simplified generate class for the pointer generator taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/generation_utils.py
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.
        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.
        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.
        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.
                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.
                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.
                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.
        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        Examples::
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            input_embeds = encoder.embed_tokens(input_ids)  * encoder.embed_scale 

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        encoder_input_ids = input_ids 

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["input_embeds"] = input_embeds
            model_kwargs["encoder_input_ids"] = encoder_input_ids

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )
            
            outputs = self(**model_inputs, return_dict=True) 
            # calling forward here 
            
            #outputs.logits (batch, seq_len, input_seq_len)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids
    
    # Function from `generation_utils.py` of Transformers library
    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


class EventGenModelWrapper(LightningModule):
    def __init__(self, bart, bart_tokenizer):
        super(EventGenModelWrapper, self).__init__()
        self.model = EventGenModel(
            BartConfig.from_pretrained("facebook/bart-base"),
            bart=bart,
            bart_tokenizer=bart_tokenizer
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(
        self, 
        encoder_input_ids, 
        encoder_attention_mask,
        dec_input_ids,
        dec_attention_mask,
        training: bool = False
    ):
        return self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=dec_attention_mask,
            training=training
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            labels.view(-1)
        )

        loss = torch.mean(loss)
    
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            labels.view(-1)
        )

        loss = torch.mean(loss)
        self.log("valid_loss", loss, prog_bar=True)
    
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"]
        )

        filled_templates = self.model.generate(
            batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"], 
            do_sample=True, 
            top_k=20, 
            top_p=0.95, 
            max_length=30, 
            num_return_sequences=1,
            num_beams=1,
            repetition_penalty=1
        )

        return (
            batch["doc_keys"], 
            filled_templates, 
            batch["dec_input_ids"]
        )

    def test_epoch_end(self, outputs):
        os.makedirs(
            "results/historical_events/events",
            exist_ok=True
        )

        with open(
            "results/historical_events/events/event_gen_predictions.jsonl",
            "w"
        ) as writer:
            for tup in outputs:
                pred = {
                    "doc_key": tup[0],
                    "predicted": self.model.tokenizer.decode(
                        tup[1].squeeze(0), 
                        skip_special_tokens=True
                    ),
                    "gold": self.model.tokenizer.decode(
                        tup[2].squeeze(0), 
                        skip_special_tokens=True
                    ) 
                }
                writer.write(json.dumps(pred)+'\n')

        return {}


class EventGenModel(PreTrainedModel):
    """Code adapted from the paper: 
    Li et al. - Document-Level Event Argument Extraction 
    by Conditional Generation, in Proceedings of the 2021 
    Conference of the North American Chapter of the 
    Association for Computational Linguistics: Human 
    Language Technologies (2021).

    https://github.com/raspberryice/gen-arg/
    
    """

    def __init__(self, config, bart, bart_tokenizer):
        super(EventGenModel, self).__init__(config)
        self.config = config
        self.tokenizer = bart_tokenizer
        self.tokenizer.add_tokens([" <arg>"," <tgr>", " <evt>"])

        self.transformer = bart
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = self.config.vocab_size = len(self.tokenizer)
        self.register_buffer(
            "final_logits_bias", 
            torch.zeros((1, self.transformer.shared.num_embeddings))
        )

        self.evt2sent, self.sent2evt, self.evt_vocab, self.evt_ids = get_event_names_dict()

    def remove_unseen(self, lm_logits, input_ids):
        # input_ids (batch, seq)
        seen_lm_logits = torch.full_like(lm_logits, fill_value=-1000).to(lm_logits.device) #(batch, seq, vocab)
        input_ids.to(lm_logits.device)
        
        seen_vocab = set(input_ids.reshape(-1).tolist())
        for i in range(self.transformer.vocab_size):
            if i in (seen_vocab):
                seen_lm_logits[:, :, i] = lm_logits[:, :, i]
        return seen_lm_logits 

    def forward(
        self, 
        input_ids, 
        attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None, # from here, parameters passed by generate
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_embeds=None,
        training: bool = False # custom parameter to discriminate training ouputs and generate outputs
    ):
        # The decoder takes the sequence, starting from
        # the special start token, and predicts the next
        # token in the sequence. The labels are thus the
        # decoder input tokens shifted one left (in other
        # words, the decoder input tokens are shifted one
        # right wrt the labels).
        labels = None
        if training:
            labels = decoder_input_ids[:, 1:].clone() 
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_attention_mask = decoder_attention_mask[:, :-1]

            # Pad tokens must be manually replaced with -100:
            # https://discuss.huggingface.co/t/is-there-a-way-to-return-the-decoder-input-ids-from-tokenizer-prepare-seq2seq-batch/2929/3
            labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask= decoder_attention_mask,
            use_cache=use_cache, 
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        decoder_output = outputs.last_hidden_state  # (batch, dec_seq_len, hidden_dim)
        encoder_output = outputs.encoder_last_hidden_state  # (batch, enc_seq_len, hidden_dim)
            
        logits = F.linear(
            decoder_output, 
            self.transformer.shared.weight, 
            bias=self.final_logits_bias
        )

        logits = self.remove_unseen(
            logits,
            self.evt_ids
        )

        """
        ls = []
        for _ in range(len(input_ids)):
            ls.append(self.evt_ids)

        batch_evt_ids = torch.stack(ls)
        batch_evt_ids = torch.tensor(batch_evt_ids, device="cuda")

        # We limit the vocabulary to that of the events.
        if input_embeds == None:  # input_embeds is passed by generate.  
            evt_tokens_emb = self.transformer.encoder.embed_tokens(
                batch_evt_ids) * self.transformer.encoder.embed_scale  # (batch, enc_seq_len, hidden_dim)
        else:
            evt_tokens_emb = input_embeds

        # Compute the formula from the paper:
        # h_i^T Emb(w)
        # We have to do this for the elements in the batch.
        # This is basically Batch Matrix Multiplication (BMM).
        # For each batch we have the probability of each of
        # the words in the encoder input (enc_seq_len) to
        # be generated by the decoder (dec_seq_len).   
        prod = torch.einsum(
            "bij,bjk->bik", 
            decoder_output, 
            torch.transpose(evt_tokens_emb, 1, 2)
        )  # (batch, dec_seq_len, enc_seq_len)

        batch_size = prod.size(0)
        dec_seq_len = prod.size(1)
        logits = torch.full(
            (
                batch_size, 
                dec_seq_len, 
                self.transformer.config.vocab_size
            ), 
            fill_value=-1000,
            dtype=prod.dtype
        ).to(prod.device)

        # Possible duplicate indices.
        index = input_ids.unsqueeze(dim=1).expand_as(prod)

        # With dim=2 this is equivalent to:
        # self[i][j][index[i][j][k]] = src[i][j][k]
        logits.scatter_(dim=2, index=index, src=prod)
        """

        if training:
            return logits, labels
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def get_encoder(self):
        return self.transformer.encoder

    def get_output_embeddings(self):
        # this method is needed for generation
        vocab_size, emb_size = self.transformer.shared.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = self.transformer.shared.weight.data
        return lin_layer 

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, input_embeds, encoder_input_ids, **kwargs):
        return {
            "input_ids": encoder_input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "input_embeds": input_embeds,
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    # # this is a simplified generate class for the pointer generator taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/generation_utils.py
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.
        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.
        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.
        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.
                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.
                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.
                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.
        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        Examples::
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            input_embeds = encoder.embed_tokens(input_ids)  * encoder.embed_scale 

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        encoder_input_ids = input_ids 

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["input_embeds"] = input_embeds
            model_kwargs["encoder_input_ids"] = encoder_input_ids

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )
            
            outputs = self(**model_inputs, return_dict=True) 
            # calling forward here 
            
            #outputs.logits (batch, seq_len, input_seq_len)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids
    
    # Function from `generation_utils.py` of Transformers library
    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


class BiLSTM_clf(LightningModule):
    def __init__(
        self,
        dropout_rate=0.2,
        num_layers=3, 
        hidden_dim=512,
        freeze_bert=False,
        average="weighted"
    ):
        super(BiLSTM_clf, self).__init__()
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained(
            "bert-base-cased"
        )

        if freeze_bert:
            for param in self.bert.named_parameters():
                param[1].requires_grad=False

        self.lstm = LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = Dropout(
            p=dropout_rate
        )
        self.linear = Linear(
            hidden_dim,
            128
        )
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(
            p=dropout_rate
        )
        self.clf = nn.Linear(128, 1)

        for param in self.linear.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.average = average

        # For binary classification we need to set
        # num_classes to 1. See:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5705
        self.accuracy = Accuracy(
            num_classes=1,
            average=self.average
        )
        self.f1 = F1(
            num_classes=1,
            average=self.average
        )
        self.prec = Precision(
            num_classes=1,
            average=self.average
        )
        self.recall = Recall(
            num_classes=1,
            average=self.average
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = output.last_hidden_state
    
        out, _ = self.lstm(
            last_hidden_state
        )
        out = torch.mean(out, -1)
        out = self.dropout1(out)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout2(out)
        logits = self.clf(out)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(
            logits.view(-1), labels
        )

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in train f1, setting to 0")
            f1 = torch.tensor(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "train_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(
            logits.view(-1), labels
        )

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in valid f1, setting to 0")
            f1 = torch.tensor(0)

        self.log("valid_loss", loss, prog_bar=True)
        self.log(
            "valid_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "valid_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in test f1, setting to 0")
            f1 = torch.tensor(0)

        self.log(
            "test_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )

        prec = self.prec(logits.view(-1), labels.long())
        recall = self.recall(logits.view(-1), labels.long())
        
        return (acc, f1, prec, recall)

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/document_classification"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item() 
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/bilstm_clf_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


class Bert_clf(LightningModule):
    def __init__(
        self,
        dropout_rate=0.2, 
        hidden_dim=128,
        freeze_bert=False,
        average="weighted"
    ):
        super(Bert_clf, self).__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-cased"
        )

        if freeze_bert:
            for param in self.bert.named_parameters():
                param[1].requires_grad=False

        self.dropout1 = Dropout(
            p=dropout_rate
        )
        self.clf = nn.Linear(768, 1)

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.average = average

        # For binary classification we need to set
        # num_classes to 1. See:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5705
        self.accuracy = Accuracy(
            num_classes=1,
            average=self.average
        )
        self.f1 = F1(
            num_classes=1,
            average=self.average
        )
        self.prec = Precision(
            num_classes=1,
            average=self.average
        )
        self.recall = Recall(
            num_classes=1,
            average=self.average
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        out = output[1]
        out = self.dropout1(out)
        logits = self.clf(out)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(
            logits.view(-1), labels
        )

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in train f1, setting to 0")
            f1 = torch.tensor(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "train_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(
            logits.view(-1), labels
        )

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in valid f1, setting to 0")
            f1 = torch.tensor(0)

        self.log("valid_loss", loss, prog_bar=True)
        self.log(
            "valid_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "valid_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        acc = self.accuracy(
            logits.view(-1), labels.long()
        )

        try:
            f1 = self.f1(
                logits.view(-1), labels.long()
            )
        except Exception as ex:
            traceback.print_exc()
            print("Error in test f1, setting to 0")
            f1 = torch.tensor(0)

        self.log(
            "test_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )

        prec = self.prec(logits.view(-1), labels.long())
        recall = self.recall(logits.view(-1), labels.long())
        
        return (acc, f1, prec, recall)

    def test_epoch_end(self, outputs):
        save_dir = "results/historical_events/document_classification"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item() 
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/bert_clf_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


        
