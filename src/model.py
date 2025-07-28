import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import numpy as np

class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super(HierarchicalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sentence_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.paragraph_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.section_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.sentence_proj = nn.Linear(hidden_size, hidden_size)
        self.paragraph_proj = nn.Linear(hidden_size, hidden_size)
        self.section_proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, embeddings: torch.Tensor, level: str = 'sentence') -> torch.Tensor:
        if level == 'sentence':
            attended, _ = self.sentence_attention(embeddings, embeddings, embeddings)
            return self.sentence_proj(attended)
        elif level == 'paragraph':
            attended, _ = self.paragraph_attention(embeddings, embeddings, embeddings)
            return self.paragraph_proj(attended)
        else:
            attended, _ = self.section_attention(embeddings, embeddings, embeddings)
            return self.section_proj(attended)

class FewShotLearner(nn.Module):
    def __init__(self, hidden_size: int, num_support: int = 5):
        super(FewShotLearner, self).__init__()
        self.hidden_size = hidden_size
        self.num_support = num_support
        self.support_encoder = nn.Linear(hidden_size, hidden_size)
        self.query_encoder = nn.Linear(hidden_size, hidden_size)
        self.distance_metric = nn.CosineSimilarity(dim=-1)
    def compute_prototype(self, support_embeddings: torch.Tensor) -> torch.Tensor:
        return support_embeddings.mean(dim=0)
    def forward(self, support_embeddings: torch.Tensor, query_embedding: torch.Tensor) -> torch.Tensor:
        prototype = self.compute_prototype(support_embeddings)
        similarity = self.distance_metric(query_embedding, prototype)
        return similarity

class ContrastiveLearner(nn.Module):
    def __init__(self, hidden_size: int, temperature: float = 0.1):
        super(ContrastiveLearner, self).__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        anchor_proj = self.projection(anchor)
        positive_proj = self.projection(positive)
        negative_proj = self.projection(negative)
        pos_sim = F.cosine_similarity(anchor_proj, positive_proj, dim=-1)
        neg_sim = F.cosine_similarity(anchor_proj, negative_proj, dim=-1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

class PersonaDocumentModel(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 max_length: int = 512, hidden_size: int = 384):
        super(PersonaDocumentModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.hierarchical_attention = HierarchicalAttention(hidden_size)
        self.few_shot_learner = FewShotLearner(hidden_size)
        self.contrastive_learner = ContrastiveLearner(hidden_size)
        self.persona_projection = nn.Linear(hidden_size, hidden_size)
        self.job_projection = nn.Linear(hidden_size, hidden_size)
        self.document_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        self.explanation_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 7)
        )
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    def forward(self, persona_text: str, job_text: str, document_text: str,
                support_examples: List[Dict] = None, contrastive_mode: bool = False) -> Dict[str, torch.Tensor]:
        persona_emb = self.encode_text(persona_text)
        job_emb = self.encode_text(job_text)
        doc_emb = self.encode_text(document_text)
        doc_hierarchical = self.hierarchical_attention(doc_emb.unsqueeze(1), 'sentence').squeeze(1)
        persona_proj = self.persona_projection(persona_emb)
        job_proj = self.job_projection(job_emb)
        doc_proj = self.document_projection(doc_hierarchical)
        persona_job_emb = (persona_proj + job_proj) / 2
        attn_output, _ = self.attention(
            persona_job_emb.unsqueeze(1), 
            doc_proj.unsqueeze(1), 
            doc_proj.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)
        few_shot_score = torch.tensor(0.0)
        if support_examples:
            support_embs = []
            for example in support_examples:
                support_emb = self.encode_text(example['document_text'])
                support_embs.append(support_emb)
            support_embs = torch.stack(support_embs)
            few_shot_score = self.few_shot_learner(support_embs, doc_emb)
        combined_features = torch.cat([persona_proj, job_proj, attn_output, doc_hierarchical], dim=1)
        relevance_score = torch.sigmoid(self.classifier(combined_features))
        explanation_features = self.explanation_head(combined_features)
        outputs = {
            'relevance_score': relevance_score,
            'explanation_features': explanation_features,
            'persona_emb': persona_emb,
            'job_emb': job_emb,
            'doc_emb': doc_emb,
            'few_shot_score': few_shot_score,
            'hierarchical_emb': doc_hierarchical
        }
        if contrastive_mode and support_examples:
            positive_emb = support_embs[0]
            negative_emb = self.encode_text("This is a completely unrelated text.")
            contrastive_loss = self.contrastive_learner(doc_emb, positive_emb, negative_emb)
            outputs['contrastive_loss'] = contrastive_loss
        return outputs
    def predict(self, persona_text: str, job_text: str, document_text: str,
                support_examples: List[Dict] = None) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(persona_text, job_text, document_text, support_examples)
            relevance_score = outputs['relevance_score'].item()
            few_shot_score = outputs['few_shot_score'].item()
            explanation_features = outputs['explanation_features'].squeeze()
            final_score = (relevance_score + few_shot_score) / 2
            explanation = self._generate_enhanced_explanation(explanation_features, few_shot_score)
            return {
                'relevance_score': final_score,
                'base_score': relevance_score,
                'few_shot_score': few_shot_score,
                'explanation': explanation
            }
    def _generate_enhanced_explanation(self, explanation_features: torch.Tensor, few_shot_score: float) -> str:
        explanations = []
        if explanation_features[0] > 0.5:
            explanations.append("High semantic similarity to persona/job")
        if explanation_features[1] > 0.5:
            explanations.append("Contains relevant keywords")
        if explanation_features[2] > 0.5:
            explanations.append("Matches persona's domain expertise")
        if explanation_features[3] > 0.5:
            explanations.append("Addresses job requirements")
        if explanation_features[4] > 0.5:
            explanations.append("Contains important entities")
        if explanation_features[5] > 0.5:
            explanations.append("Hierarchical attention indicates importance")
        if explanation_features[6] > 0.5:
            explanations.append("Strong structural relevance")
        if few_shot_score > 0.7:
            explanations.append("Similar to high-relevance examples")
        if not explanations:
            explanations.append("General relevance to persona/job")
        return "; ".join(explanations) 