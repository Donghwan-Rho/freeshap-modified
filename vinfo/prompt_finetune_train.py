"""
Prompt-based Fine-tuning for BERT on SST-2
Matching FreeShap configuration exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import os


class PromptSST2Dataset(Dataset):
    """
    SST-2 dataset with prompt template applied
    Template: "*cls**sent_0*_It_was*mask*.*sep+*"
    """
    def __init__(self, examples, tokenizer, label_word_list, max_length=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_word_list = label_word_list
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        sentence = example['sentence']
        label = example['label']
        
        # Apply prompt template: "sentence It was [MASK]."
        prompted_text = f"{sentence} It was {self.tokenizer.mask_token}."
        
        # Tokenize
        encoded = self.tokenizer(
            prompted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find [MASK] position
        mask_positions = (encoded['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_pos = mask_positions[0].item() if len(mask_positions) > 0 else 0
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'mask_pos': mask_pos,
            'label': label
        }


class PromptBERTForSST2(nn.Module):
    """
    BERT with prompt-based fine-tuning for SST-2
    Matches FreeShap's PromptFinetuneProbe configuration
    """
    def __init__(self, 
                 model_name='bert-base-uncased',
                 label_word_list=None,
                 num_frozen_layers=8,
                 num_labels=2):
        super().__init__()
        
        # Load BERT config and disable dropout for deterministic behavior
        config = BertConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        
        # Load BERT with MaskedLM head
        self.bert = BertForMaskedLM.from_pretrained(model_name, config=config)
        self.config = config
        self.num_labels = num_labels
        self.label_word_list = label_word_list
        self.model_name = model_name
        
        # Freeze first num_frozen_layers encoder layers
        for layer in self.bert.bert.encoder.layer[:num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze pooler if exists
        if hasattr(self.bert.bert, 'pooler') and self.bert.bert.pooler is not None:
            for param in self.bert.bert.pooler.parameters():
                param.requires_grad = False
        
        # Reduce MLM head to only label words
        if label_word_list is not None:
            self._reduce_to_label_words()
        
        self.print_param_count()
        
    def _reduce_to_label_words(self):
        """
        Reduce MaskedLM head output from vocab_size to num_labels
        Only keep embeddings for label words (terrible, great)
        """
        # Extract original weights
        original_decoder = self.bert.cls.predictions.decoder
        original_bias = self.bert.cls.predictions.bias
        
        # Get label word embeddings
        label_word_embeddings = self.bert.bert.embeddings.word_embeddings.weight[self.label_word_list, :]
        label_word_bias = original_bias.data[self.label_word_list]
        
        # Create new smaller decoder
        hidden_size = self.config.hidden_size
        self.bert.cls.predictions.decoder = nn.Linear(hidden_size, self.num_labels, bias=True)
        self.bert.cls.predictions.decoder.weight = nn.Parameter(label_word_embeddings)
        self.bert.cls.predictions.decoder.bias = nn.Parameter(label_word_bias)
        
        # Freeze the bias (following FreeShap implementation)
        self.bert.cls.predictions.bias.requires_grad = False
    
    def print_param_count(self):
        """Print parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Frozen parameters: {total_params - trainable_params:,}')
        print(f'Trainable ratio: {trainable_params/total_params*100:.2f}%')
    
    def forward(self, input_ids, attention_mask, token_type_ids, mask_pos):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            mask_pos: [batch_size] - position of [MASK] token
        
        Returns:
            logits: [batch_size, num_labels]
        """
        # Forward through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get logits: [batch_size, seq_len, num_labels]
        logits = outputs.logits
        
        # Extract logits at [MASK] position
        batch_size = logits.size(0)
        if isinstance(mask_pos, int):
            mask_logits = logits[:, mask_pos, :]
        else:
            mask_logits = logits[torch.arange(batch_size), mask_pos, :]
        
        return mask_logits


def load_indices_from_file(file_path):
    """Load indices from txt file
    
    Args:
        file_path: Path to txt file containing indices
        Format can be:
        - One index per line
        - Comma-separated indices
        - Space-separated indices
    
    Returns:
        List of indices
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Try to parse as comma-separated
    if ',' in content:
        indices = [int(idx.strip()) for idx in content.split(',') if idx.strip()]
    # Try to parse as space-separated
    elif ' ' in content:
        indices = [int(idx.strip()) for idx in content.split() if idx.strip()]
    # Parse as newline-separated
    else:
        indices = [int(idx.strip()) for idx in content.split('\n') if idx.strip()]
    
    return indices


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        mask_pos = batch['mask_pos'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask, token_type_ids, mask_pos)
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}'
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def evaluate(model, val_loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            mask_pos = batch['mask_pos'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids, mask_pos)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Prompt-based Fine-tuning for BERT on SST-2')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='BERT model name')
    parser.add_argument('--num_train_dp', type=int, default=10000,
                        help='Number of training data points')
    parser.add_argument('--train_start_idx', type=int, default=0,
                        help='Start index for training data (inclusive)')
    parser.add_argument('--train_end_idx', type=int, default=None,
                        help='End index for training data (exclusive, None means use all)')
    parser.add_argument('--train_indices_file', type=str, default=None,
                        help='Path to txt file containing train indices (overrides start/end idx)')
    parser.add_argument('--train_data_percentage', type=float, default=100.0,
                        help='Percentage of data to use from indices file (0-100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay')
    parser.add_argument('--num_frozen_layers', type=int, default=8,
                        help='Number of frozen encoder layers')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("Prompt-based Fine-tuning for BERT on SST-2")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: SST-2")
    print(f"Training samples: {args.num_train_dp}")
    print(f"Frozen layers: {args.num_frozen_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get label word IDs
    # Following FreeShap config: {'0':'terrible','1':'great'}
    label_word_list = [
        tokenizer.convert_tokens_to_ids("terrible"),  # negative (class 0)
        tokenizer.convert_tokens_to_ids("great")      # positive (class 1)
    ]
    print(f"Label words: terrible (ID: {label_word_list[0]}), great (ID: {label_word_list[1]})")
    
    # Load dataset
    print("\n[2/6] Loading SST-2 dataset...")
    dataset = load_dataset("sst2")
    
    # Prepare train set
    if args.train_indices_file is not None:
        # Load indices from file
        print(f"Loading train indices from: {args.train_indices_file}")
        train_indices = load_indices_from_file(args.train_indices_file)
        print(f"Loaded {len(train_indices)} indices")
        
        # Apply percentage filter
        if args.train_data_percentage < 100.0:
            num_to_use = int(len(train_indices) * args.train_data_percentage / 100.0)
            train_indices = train_indices[:num_to_use]
            print(f"Using {args.train_data_percentage}% of data: {len(train_indices)} indices")
        
        # Select examples by indices
        all_train_examples = list(dataset['train'])
        train_examples = [all_train_examples[idx] for idx in train_indices]
        print(f"Train examples: {len(train_examples)} (from indices file)")
    else:
        # Use index slicing
        train_end_idx = args.train_end_idx if args.train_end_idx is not None else args.num_train_dp
        train_examples = list(dataset['train'])[args.train_start_idx:train_end_idx]
        print(f"Train examples: {len(train_examples)} (indices {args.train_start_idx}:{train_end_idx})")
    
    val_examples = list(dataset['validation'])
    print(f"Validation examples: {len(val_examples)}")
    
    # Create datasets
    train_dataset = PromptSST2Dataset(train_examples, tokenizer, label_word_list, args.max_length)
    val_dataset = PromptSST2Dataset(val_examples, tokenizer, label_word_list, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\n[3/6] Creating model...")
    model = PromptBERTForSST2(
        model_name=args.model_name,
        label_word_list=label_word_list,
        num_frozen_layers=args.num_frozen_layers,
        num_labels=2
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer (following FreeShap: Adam with weight_decay)
    print("\n[4/6] Creating optimizer...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: Adam(lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Training loop
    print("\n[5/6] Training...")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.max_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.max_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, 'best_prompt_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    # Final evaluation
    print("\n[6/6] Final Evaluation...")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}")
    
    # Load best model and evaluate
    # checkpoint = torch.load(os.path.join(args.save_dir, 'best_prompt_model.pt'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = evaluate(model, val_loader, device)
    print(f"Final Test Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    # print(f"\nModel saved to: {args.save_dir}/best_prompt_model.pt")


if __name__ == '__main__':
    main()
