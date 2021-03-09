import torch
import torch.nn as nn

from transformers import *

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaProtestAuxClassification(DistilBertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaProtestAuxClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.aux_classes = config.aux_classes

        self.roberta = RobertaModel(config)
        self.pre_enc = nn.Linear(config.hidden_size, config.hidden_size)
        self.label_clf = nn.Linear(config.hidden_size, self.num_labels)

        self.aux_enc = nn.Linear(config.hidden_size + self.num_labels,
                config.hidden_size + self.num_labels)
        self.aux_clf = nn.Linear(config.hidden_size + self.num_labels,
                self.aux_classes)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = roberta_output[0]              # (bs, seq_len, dim)
        transformer_out = hidden_state[:, 0]          # (bs, dim)

        pooled_output = self.pre_enc(self.dropout(transformer_out))
        label_output = torch.tanh(pooled_output)

        label_logits = self.dropout(label_output)
        label_logits = self.label_clf(label_logits)
        aux_inputs = torch.cat((self.dropout(transformer_out), label_logits), dim=-1)

        aux_output = torch.tanh(self.aux_enc(aux_inputs))
        aux_logits = self.aux_clf(self.dropout(aux_output))

        outputs = (label_logits, aux_logits,)
        outputs = outputs + roberta_output[1:]
        if labels is not None:
            labels, aux = labels
            loss_fct = nn.CrossEntropyLoss()

            label_loss = loss_fct(label_logits.view(-1, self.num_labels), labels.long().view(-1))
            aux_loss = loss_fct(aux_logits.view(-1, self.aux_classes), aux.long().view(-1))

            loss = label_loss + aux_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), label_logits, aux_logits, (hidden_states), (attentions)


class RobertaMultiTaskClassification(DistilBertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaMultiTaskClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.form_classes = config.form_classes
        self.issue_classes = config.issue_classes
        self.target_classes = config.target_classes

        self.roberta = RobertaModel(config)
        self.pre_enc = nn.Linear(config.hidden_size, config.hidden_size)
        self.label_clf = nn.Linear(config.hidden_size, self.num_labels)

        self.form_enc = nn.Linear(config.hidden_size + self.num_labels, config.hidden_size)
        self.form_clf = nn.Linear(config.hidden_size, self.form_classes)

        self.issue_enc = nn.Linear(config.hidden_size + self.num_labels, config.hidden_size)
        self.issue_clf = nn.Linear(config.hidden_size, self.issue_classes)

        self.target_enc = nn.Linear(config.hidden_size + self.num_labels, config.hidden_size)
        self.target_clf = nn.Linear(config.hidden_size, self.target_classes)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = roberta_output[0]              # (bs, seq_len, dim)
        transformer_out = hidden_state[:, 0]          # (bs, dim)

        pooled_output = self.pre_enc(self.dropout(transformer_out))
        label_output = torch.tanh(pooled_output)

        label_logits = self.dropout(label_output)
        label_logits = self.label_clf(label_logits)
        aux_inputs = torch.cat((self.dropout(transformer_out), label_logits), dim=-1)

        form_output = torch.tanh(self.form_enc(aux_inputs))
        form_logits = self.form_clf(self.dropout(form_output))

        issue_output = torch.tanh(self.issue_enc(aux_inputs))
        issue_logits = self.issue_clf(self.dropout(issue_output))

        target_output = torch.tanh(self.target_enc(aux_inputs))
        target_logits = self.target_clf(self.dropout(target_output))

        outputs = (label_logits, form_logits, issue_logits, target_logits,)
        outputs = outputs + roberta_output[1:]
        if labels is not None:
            labels, form, issue, target = labels
            loss_fct = nn.CrossEntropyLoss()

            label_loss = loss_fct(label_logits.view(-1, self.num_labels),
                    labels.view(-1))
            form_loss = loss_fct(form_logits.view(-1, self.form_classes),
                    form.view(-1))
            issue_loss = loss_fct(issue_logits.view(-1, self.issue_classes),
                    issue.view(-1))
            target_loss = loss_fct(target_logits.view(-1, self.target_classes),
                    target.view(-1))

            loss = label_loss + (form_loss + issue_loss + target_loss).mean()
            outputs = (loss,) + outputs

        return outputs  # (loss), label_logits, form_logits, issue_logits, target_logits, (hidden_states), (attentions)
