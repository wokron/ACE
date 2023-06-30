from pathlib import Path
from typing import Union, List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import relu

import flair
from flair.data import DataPoint, Dictionary, Sentence
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result


class TransE(flair.nn.Model):
    def __init__(
            self,
            hidden_size: int,
            entity_num_embeddings: int,
            embeddings: TokenEmbeddings,
            # tag_type: str,
            target_languages: int = 1,
            sentence_loss: bool = False,
            dropout: float = 0.0,
            config=None,
            L=2,
            margin=1.0
    ):
        super(TransE, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.target_languages: int = target_languages
        self.sentence_level_loss = sentence_loss
        self.entity_num_embeddings = entity_num_embeddings
        self.dropout = dropout
        self.config = config
        self.tag_type = "kgc"
        self.use_crf = False
        self.use_decoder_timer = False
        self.tag_dictionary = Dictionary()
        self.use_language_attention = False

        self.entity_embedding = torch.nn.Embedding(self.entity_num_embeddings, self.embeddings.embedding_length).to(
            flair.device)
        self.dist_fn = torch.nn.PairwiseDistance(L).to(flair.device)
        self.margin = margin
        self.L = L

    def score(self, head, relation, tail):
        """
        Args:
            head: Tensor(batch_size, embed_dim)
            relation: Tensor(batch_size, embed_dim)
            tail: Tensor(batch_size, embed_dim)

        Returns:
            the distance between head + relation and tail
        """
        return self.dist_fn(head + relation, tail)

    def forward(self, relation: List[Sentence], pos: (Tensor, Tensor), neg: (Tensor, Tensor)):
        """
        Args:
            relation: List(batch_size,)
            pos: Tensor(batch_size,), Tensor(batch_size,)
            neg: Tensor(batch_size,), Tensor(batch_size,)

        Returns:
            the loss of TransE model
        """
        batch_size = len(relation)

        sentences = relation
        self.embeddings.embed(sentences)
        sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] for idx, x in
                                     enumerate(sorted(sentences.features.keys()))],
                                    -1)  # (batch_size, seq_len, embed_dim)

        relation_tensor = torch.mean(sentence_tensor, dim=1)  # (batch_size, embed_dim)

        pos_head_tensor = self.entity_embedding(pos[0])
        pos_tail_tensor = self.entity_embedding(pos[1])
        pos_score = self.score(pos_head_tensor, relation_tensor, pos_tail_tensor)

        neg_head_tensor = self.entity_embedding(neg[0])
        neg_tail_tensor = self.entity_embedding(neg[1])
        neg_score = self.score(neg_head_tensor, relation_tensor, neg_tail_tensor)

        return torch.sum(relu(input=pos_score - neg_score + self.margin)) / batch_size

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        if type(sentences) == Sentence:
            sentences = [sentences]

        batch_size = len(sentences)

        head_id_list, tail_id_list = zip(*[(sentence.head_id, sentence.tail_id) for sentence in sentences])

        actual_head_tensor = torch.tensor(head_id_list, device=flair.device)
        actual_tail_tensor = torch.tensor(tail_id_list, device=flair.device)

        pos_tensor = (actual_head_tensor, actual_tail_tensor)

        head_or_tail = torch.randint(high=2, size=[batch_size], device=flair.device)
        random_entities = torch.randint(high=self.entity_num_embeddings, size=[batch_size], device=flair.device)

        false_head_tensor = torch.where(head_or_tail == 1, random_entities, actual_head_tensor)
        false_tail_tensor = torch.where(head_or_tail == 0, random_entities, actual_tail_tensor)

        neg_tensor = (false_head_tensor, false_tail_tensor)

        loss = self.forward(sentences, pos_tensor, neg_tensor)

        return loss

    def evaluate(
            self,
            data_loader: DataLoader,
            out_path: Path = None,
            embeddings_storage_mode: str = "cpu",
            prediction_mode=False,
            speed_test=False,
    ) -> (Result, float):
        detailed_result = (
            "\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
            "\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
        )

        result = Result(
            main_score=0.5,
            log_line=f"finish evaluate",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, torch.tensor(0.5)  # todo: mock evaluate

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "entity_num_embeddings": self.entity_num_embeddings,
            "embeddings": self.embeddings,
            "target_languages": self.target_languages,
            "sentence_loss": self.sentence_level_loss,
            "dropout": self.dropout,
            "config": self.config,
            "L": self.L,
            "margin": self.margin,

            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
        }
        return model_state

    def _init_model_with_state_dict(state, testing=False):
        model = TransE(
            hidden_size=state['hidden_size'],
            entity_num_embeddings=state['entity_num_embeddings'],
            embeddings=state['embeddings'],
            target_languages=state['target_languages'],
            sentence_loss=state['sentence_loss'],
            dropout=state['dropout'],
            config=state['config'],
            L=state['L'],
            margin=state['margin'],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def _fetch_model(model_name) -> str:
        return model_name

    def get_state(self, ):
        return None
