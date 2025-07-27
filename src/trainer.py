import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json
import os
from src.model import PersonaDocumentModel

class EnhancedPersonaDocumentDataset(Dataset):
    """
    Enhanced dataset for training the persona-document model with few-shot examples.
    """
    
    def __init__(self, data: List[Dict], support_examples: List[Dict] = None):
        self.data = data
        self.support_examples = support_examples or []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'persona_text': item['persona_text'],
            'job_text': item['job_text'],
            'document_text': item['document_text'],
            'relevance_score': torch.tensor(item['relevance_score'], dtype=torch.float32),
            'explanation': item['explanation'],
            'support_examples': self.support_examples
        }

class EnhancedModelTrainer:
    """
    Enhanced trainer for the PersonaDocumentModel with few-shot and contrastive learning.
    """
    
    def __init__(self, model: PersonaDocumentModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.contrastive_weight = 0.1  # Weight for contrastive loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with enhanced features."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass with few-shot learning
            outputs = self.model(
                batch['persona_text'],
                batch['job_text'],
                batch['document_text'],
                support_examples=batch.get('support_examples', None),
                contrastive_mode=True
            )
            
            # Fix tensor shape for binary cross entropy
            relevance_scores = outputs['relevance_score'].squeeze()  # Remove extra dimension
            target_scores = batch['relevance_score'].to(self.device)
            
            # Calculate main loss
            main_loss = self.criterion(relevance_scores, target_scores)
            
            # Calculate contrastive loss if available
            contrastive_loss = torch.tensor(0.0)
            if 'contrastive_loss' in outputs:
                contrastive_loss = outputs['contrastive_loss']
            
            # Combined loss
            total_loss_batch = main_loss + self.contrastive_weight * contrastive_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    batch['persona_text'],
                    batch['job_text'],
                    batch['document_text'],
                    support_examples=batch.get('support_examples', None)
                )
                
                # Fix tensor shape for binary cross entropy
                relevance_scores = outputs['relevance_score'].squeeze()  # Remove extra dimension
                target_scores = batch['relevance_score'].to(self.device)
                
                loss = self.criterion(relevance_scores, target_scores)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def generate_enhanced_synthetic_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Generate enhanced synthetic training data with few-shot examples.
    """
    # Main training data
    main_data = [
        {
            'persona_text': 'PhD Researcher in Computational Biology',
            'job_text': 'Prepare a comprehensive literature review focusing on methodologies',
            'document_text': 'The methodology section describes the experimental setup and data collection procedures.',
            'relevance_score': 0.9,
            'explanation': 'High semantic similarity to persona/job; Contains relevant keywords'
        },
        {
            'persona_text': 'Investment Analyst',
            'job_text': 'Analyze revenue trends and market positioning',
            'document_text': 'The financial results show a 15% increase in revenue compared to last year.',
            'relevance_score': 0.8,
            'explanation': 'Contains relevant keywords; Addresses job requirements'
        },
        {
            'persona_text': 'Undergraduate Chemistry Student',
            'job_text': 'Identify key concepts for exam preparation',
            'document_text': 'The reaction mechanism involves a series of intermediate steps.',
            'relevance_score': 0.7,
            'explanation': 'Matches persona domain expertise; Contains relevant keywords'
        }
    ]
    
    # Few-shot support examples
    support_examples = [
        {
            'persona_text': 'PhD Researcher in Computational Biology',
            'job_text': 'Prepare a comprehensive literature review focusing on methodologies',
            'document_text': 'The experimental methodology includes advanced statistical analysis techniques.',
            'relevance_score': 0.95,
            'explanation': 'Excellent match for methodology-focused research'
        },
        {
            'persona_text': 'Investment Analyst',
            'job_text': 'Analyze revenue trends and market positioning',
            'document_text': 'Quarterly revenue analysis reveals consistent growth patterns.',
            'relevance_score': 0.85,
            'explanation': 'Strong financial analysis content'
        }
    ]
    
    # Generate variations
    enhanced_main_data = []
    for sample in main_data:
        enhanced_main_data.append(sample)
        
        # Add negative samples
        negative_sample = sample.copy()
        negative_sample['document_text'] = 'This section contains general background information.'
        negative_sample['relevance_score'] = 0.2
        negative_sample['explanation'] = 'General relevance to persona/job'
        enhanced_main_data.append(negative_sample)
    
    return enhanced_main_data, support_examples

def train_enhanced_model(epochs: int = 10, batch_size: int = 4) -> PersonaDocumentModel:
    """
    Train the enhanced PersonaDocumentModel.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # Initialize model
    model = PersonaDocumentModel()
    trainer = EnhancedModelTrainer(model)
    
    # Generate enhanced synthetic data
    main_data, support_examples = generate_enhanced_synthetic_data()
    dataset = EnhancedPersonaDocumentDataset(main_data, support_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print("Starting enhanced model training...")
    print("Features: Few-shot learning, hierarchical attention, contrastive learning")
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
    
    # Save the trained model
    trainer.save_model('enhanced_persona_document_model.pth')
    print("Enhanced model saved to enhanced_persona_document_model.pth")
    
    return model 