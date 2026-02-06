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
import random
import argparse
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Suppress transformers warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)


class PromptDataset(Dataset):
    """
    Universal prompt dataset for SST2/MR/RTE/MNLI/MRPC
    Handles both single sentence and sentence pair tasks
    """
    def __init__(self, examples, tokenizer, label_word_list, max_length=64, dataset_name='sst2'):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_word_list = label_word_list
        self.max_length = max_length
        self.dataset_name = dataset_name
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        label = example['label']
        
        # Apply prompt template based on dataset
        if self.dataset_name == 'sst2':
            # SST2: "sentence It was [MASK]."
            sentence = example['sentence']
            prompted_text = f"{sentence} It was {self.tokenizer.mask_token}."
        elif self.dataset_name == 'mr':
            # MR (rotten_tomatoes): "text It was [MASK]."
            sentence = example['text']  # MR uses 'text' field, not 'sentence'
            prompted_text = f"{sentence} It was {self.tokenizer.mask_token}."
        elif self.dataset_name in ['rte', 'mnli', 'mrpc']:
            # Sentence pair: "sentence1 ? [MASK], sentence2"
            sentence1 = example['sentence1']
            sentence2 = example['sentence2']
            prompted_text = f"{sentence1} ? {self.tokenizer.mask_token}, {sentence2}"
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
        
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
    BERT with prompt-based fine-tuning
    Supports SST2/MR/RTE/MNLI/MRPC datasets
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
        # Suppress output in sweep mode
        return
        # total_params = sum(p.numel() for p in self.parameters())
        # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f'Total parameters: {total_params:,}')
        # print(f'Trainable parameters: {trainable_params:,}')
        # print(f'Frozen parameters: {total_params - trainable_params:,}')
        # print(f'Trainable ratio: {trainable_params/total_params*100:.2f}%')
    
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
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=True)
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
        for batch in tqdm(val_loader, desc='Evaluating', disable=True):
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
    parser = argparse.ArgumentParser(description='Prompt-based Fine-tuning for BERT')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='BERT model name')
    
    # Arguments for automatic path construction (matching sst2_shapley_acc.py)
    parser.add_argument('--dataset_name', type=str, default='rte',
                        help='Dataset name')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed (also used in file path)')
    parser.add_argument('--num_train_dp', type=int, default=10000,
                        help='Number of training data points')
    parser.add_argument('--val_sample_num', type=int, default=1000,
                        help='Number of validation samples (for file path)')
    parser.add_argument('--tmc_iter', type=int, default=500,
                        help='TMC iteration count (for file path)')
    parser.add_argument('--tmc_seed', type=int, default=2023,
                        help='TMC seed (for file path)')
    parser.add_argument('--approximate', type=str, default='inv', choices=['inv', 'eigen'],
                        help='Approximation method: inv or eigen')
    parser.add_argument('--eigen_rank', type=int, default=10,
                        help='Eigen rank as percentage of num_train_dp (e.g., 10 means 10%%)')
    parser.add_argument('--lambda_', type=float, default=1e-6,
                        help='Lambda (regularization parameter)')
    parser.add_argument('--train_start_idx', type=int, default=0,
                        help='Start index for training data (inclusive)')
    parser.add_argument('--train_end_idx', type=int, default=None,
                        help='End index for training data (exclusive, None means use all)')
    parser.add_argument('--train_indices_file', type=str, default=None,
                        help='Path to txt file containing train indices (overrides start/end idx)')
    parser.add_argument('--train_data_percentage', type=float, default=100.0,
                        help='Percentage of data to use from indices file (0-100)')
    parser.add_argument('--sweep_output_file', type=str, default=None,
                        help='Output file to save sweep results (auto-enabled if sweep params are set)')
    parser.add_argument('--sweep_mode', type=str, default='percentage', choices=['percentage', 'count'],
                        help="Sweep mode: 'percentage' (default, 1-100%%) or 'count' (absolute sample count)")
    parser.add_argument('--sweep_start', type=float, default=None,
                        help='Start value for sweep (percentage or count, default: 1 for percentage, 1 for count)')
    parser.add_argument('--sweep_end', type=float, default=None,
                        help='End value for sweep (percentage or count, default: 100 for percentage, total_samples for count)')
    parser.add_argument('--sweep_step', type=float, default=None,
                        help='Step size for sweep (percentage or count, default: 1 for percentage, 1 for count)')
    parser.add_argument('--select_indices', type=str, default=None,
                        help='Comma-separated list of indices to select from loaded train_indices (e.g., "0,6,10")')
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
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Construct train_indices_file path automatically if not provided
    if args.train_indices_file is None:
        # Calculate actual eigen rank from percentage (for internal use only)
        # But use the percentage value itself in the filename
        
        # Determine model name from model_name
        if 'bert' in args.model_name.lower():
            model_name = 'bert'
        elif 'roberta' in args.model_name.lower():
            model_name = 'roberta'
        elif 'llama' in args.model_name.lower():
            model_name = 'llama'
        else:
            model_name = 'model'
        
        # Construct path based on approximate method
        base_path = f"./freeshap_res/{args.dataset_name}/shapley/{args.approximate}/indices"
        
        # Format lambda in scientific notation for filename (always use 1e-X format)
        lambda_str = f"{args.lambda_:.0e}"
        
        if args.approximate == 'eigen':
            # Use the percentage value itself (eigen_rank) in filename, not the calculated rank
            extra_tag = f"_eig{args.eigen_rank}_lam{lambda_str}_cholesky_float32"
        else:
            extra_tag = f"_lam{lambda_str}"
        
        args.train_indices_file = (
            f"{base_path}/{model_name}"
            f"_seed{args.seed}_num{args.num_train_dp}_val{args.val_sample_num}"
            f"{extra_tag}_signFalse_earlystopTrue"
            f"_tmc{args.tmc_seed}_iter{args.tmc_iter}_indices.txt"
        )
        print(f"[info] Auto-constructed train_indices_file: {args.train_indices_file}")
    
    # Check if sweep mode is enabled (if any sweep parameter is specified)
    sweep_enabled = (args.sweep_start is not None or args.sweep_end is not None or args.sweep_step is not None)
    
    # Construct sweep_output_file path automatically if needed
    if sweep_enabled and args.sweep_output_file is None:
        # Use pbft_acc directory for output
        base_dir = args.train_indices_file.replace('/indices/', '/pbft_acc/')
        args.sweep_output_file = base_dir.replace('_indices.txt', '_pbft_acc.txt')
        print(f"[info] Auto-constructed sweep_output_file: {args.sweep_output_file}")
    
    # Execute sweep mode if enabled
    if sweep_enabled:
        if args.train_indices_file is None:
            raise ValueError("--train_indices_file must be specified when using sweep mode")
        if args.sweep_output_file is None:
            raise ValueError("--sweep_output_file must be specified when using sweep mode")
        
        # Load full indices to get total count
        full_train_indices = load_indices_from_file(args.train_indices_file)
        total_indices = len(full_train_indices)
        
        # Determine sweep parameters based on mode
        if args.sweep_mode == 'percentage':
            sweep_start = args.sweep_start if args.sweep_start is not None else 1.0
            sweep_end = args.sweep_end if args.sweep_end is not None else 100.0
            sweep_step = args.sweep_step if args.sweep_step is not None else 1.0
            sweep_values = []
            current = sweep_start
            while current <= sweep_end:
                sweep_values.append(current)
                current += sweep_step
            mode_str = f"percentages {sweep_start}% to {sweep_end}% (step {sweep_step}%)"
        elif args.sweep_mode == 'count':
            sweep_start = int(args.sweep_start) if args.sweep_start is not None else 1
            sweep_end = int(args.sweep_end) if args.sweep_end is not None else total_indices
            sweep_step = int(args.sweep_step) if args.sweep_step is not None else 1
            sweep_values = list(range(sweep_start, sweep_end + 1, sweep_step))
            mode_str = f"counts {sweep_start} to {sweep_end} samples (step {sweep_step})"
        
        print("="*80)
        print(f"SWEEP MODE: Running experiments for {mode_str}")
        print("="*80)
        
        # Parse configuration from train_indices_file name
        import os
        import re
        filename = os.path.basename(args.train_indices_file)
        
        # Extract key parameters from filename
        config_info = {}
        
        # Dataset (e.g., sst2, mnli, etc.)
        dataset_match = re.search(r'^([a-z0-9]+)_', filename)
        if dataset_match:
            config_info['dataset'] = dataset_match.group(1)
        
        # Number of data points
        num_match = re.search(r'num(\d+)', filename)
        if num_match:
            config_info['num_dp'] = num_match.group(1)
        
        # Seed
        seed_match = re.search(r'seed(\d+)', filename)
        if seed_match:
            config_info['seed'] = seed_match.group(1)
        
        # Method (eigen vs inv)
        if 'approeigen' in filename:
            config_info['method'] = 'Eigen'
            # Rank (eig)
            eig_match = re.search(r'eig(\d+)', filename)
            if eig_match:
                config_info['rank'] = eig_match.group(1)
            # Lambda
            lam_match = re.search(r'lam([\d.]+)', filename)
            if lam_match:
                config_info['lambda'] = lam_match.group(1)
        elif 'approinv' in filename:
            config_info['method'] = 'INV'
        
        # Iteration
        iter_match = re.search(r'iter(\d+)', filename)
        if iter_match:
            config_info['iteration'] = iter_match.group(1)
        
        # Print configuration
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset_name}")
        print(f"  Method: {args.approximate.upper()}")
        if args.approximate == 'eigen':
            print(f"  Eigen Rank: {args.eigen_rank}%")
        print(f"  Lambda: {args.lambda_}")
        print(f"  Num DP: {args.num_train_dp}")
        print(f"  Seed: {args.seed}")
        print(f"  TMC Iteration: {args.tmc_iter}")
        print(f"  Total available samples: {total_indices}")
        print(f"  Model: {args.model_name}")
        print(f"  Max epochs: {args.max_epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Frozen layers: {args.num_frozen_layers}")
        print(f"  Output file: {args.sweep_output_file}")
        print(f"  Sweep mode: {args.sweep_mode}")
        if args.sweep_mode == 'percentage':
            print(f"  Sweep range: {sweep_start}% to {sweep_end}% (step {sweep_step}%)")
        else:
            print(f"  Sweep range: {sweep_start} to {sweep_end} samples (step {sweep_step})")
        print("="*80)
        
        results = []
        
        # For count mode, track previous num_samples to show newly added labels
        prev_num_samples = 0
        
        for idx, value in enumerate(sweep_values, 1):
            if args.sweep_mode == 'percentage':
                # Percentage mode
                pct = value
                num_samples = int(total_indices * pct / 100.0)
                args.train_data_percentage = float(pct)
                value_str = f"{pct:6.2f}%"
            else:
                # Count mode
                num_samples = int(value)
                pct = (num_samples / total_indices) * 100.0
                args.train_data_percentage = pct
                value_str = f"{num_samples:5d} samples ({pct:5.2f}%)"
            
            # Run single experiment
            final_acc = run_single_experiment(args)
            
            # Convert to integer format (multiply by 10000)
            acc_int = int(round(final_acc * 10000))
            results.append(acc_int)
            
            if args.sweep_mode == 'percentage':
                print(f"[{idx:3d}/{len(sweep_values):3d}] Pct {pct:6.2f}% ({num_samples:5d} samples) -> Acc: {acc_int} ({final_acc:.4f})")
            else:
                # Count mode: show newly added sample labels
                newly_added_labels = []
                if num_samples > prev_num_samples:
                    # Get labels of newly added samples
                    from datasets import load_dataset
                    
                    # Load dataset based on dataset name
                    if args.dataset_name == 'sst2':
                        dataset = load_dataset('sst2')
                    elif args.dataset_name in ['rte', 'mnli', 'mrpc']:
                        dataset = load_dataset('glue', args.dataset_name)
                    elif args.dataset_name == 'mr':
                        dataset = load_dataset('rotten_tomatoes')
                    else:
                        dataset = load_dataset(args.dataset_name)
                    
                    all_train_examples = list(dataset['train'])
                    
                    for i in range(prev_num_samples, num_samples):
                        if i < len(full_train_indices):
                            sample_idx = full_train_indices[i]
                            if sample_idx < len(all_train_examples):
                                label = all_train_examples[sample_idx]['label']
                                newly_added_labels.append(label)
                
                labels_str = f" [+{newly_added_labels}]" if newly_added_labels else ""
                print(f"[{idx:3d}/{len(sweep_values):3d}] Count {num_samples:5d} ({pct:6.2f}%) -> Acc: {acc_int} ({final_acc:.4f}){labels_str}")
                prev_num_samples = num_samples
        
        # Save results to file
        os.makedirs(os.path.dirname(args.sweep_output_file), exist_ok=True)
        with open(args.sweep_output_file, 'w') as f:
            f.write(str(results))
        
        print(f"\n{'='*80}")
        print(f"Completed all {len(sweep_values)} experiments!")
        print(f"Results saved to: {args.sweep_output_file}")
        print(f"{'='*80}\n")
        print("Final results:")
        print(results)
        
        return
    
    # Normal single-run mode
    final_acc = run_single_experiment(args)
    print(f"Final Accuracy: {final_acc:.4f}")


def run_single_experiment(args):
    """Run a single fine-tuning experiment and return final accuracy"""
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # print("="*80)
    # print("Prompt-based Fine-tuning for BERT on SST-2")
    # print("="*80)
    # print(f"Model: {args.model_name}")
    # print(f"Dataset: SST-2")
    # print(f"Training samples: {args.num_train_dp}")
    # print(f"Frozen layers: {args.num_frozen_layers}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Learning rate: {args.lr}")
    # print(f"Max epochs: {args.max_epochs}")
    # print(f"Device: {args.device}")
    # print("="*80)
    
    # Load tokenizer
    # print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get label word IDs
    # Following FreeShap config: {'0':'terrible','1':'great'}
    label_word_list = [
        tokenizer.convert_tokens_to_ids("terrible"),  # negative (class 0)
        tokenizer.convert_tokens_to_ids("great")      # positive (class 1)
    ]
    # print(f"Label words: terrible (ID: {label_word_list[0]}), great (ID: {label_word_list[1]})")
    
    # Load dataset
    # print("\n[2/6] Loading dataset...")
    if args.dataset_name == 'sst2':
        dataset = load_dataset('sst2')
    elif args.dataset_name in ['rte', 'mnli', 'mrpc']:
        dataset = load_dataset('glue', args.dataset_name)
    elif args.dataset_name == 'mr':
        dataset = load_dataset('rotten_tomatoes')
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Prepare train set
    if args.train_indices_file is not None:
        # Load indices from file
        # print(f"Loading train indices from: {args.train_indices_file}")
        train_indices = load_indices_from_file(args.train_indices_file)
        # print(f"Loaded {len(train_indices)} indices")
        
        # Apply select_indices filter first (if specified)
        if hasattr(args, 'select_indices') and args.select_indices is not None:
            selected_positions = [int(x.strip()) for x in args.select_indices.split(',')]
            train_indices = [train_indices[pos] for pos in selected_positions if pos < len(train_indices)]
            # print(f"Selected indices at positions {selected_positions}: {len(train_indices)} indices")
        # Apply percentage filter (if select_indices not used)
        elif args.train_data_percentage < 100.0:
            num_to_use = int(len(train_indices) * args.train_data_percentage / 100.0)
            train_indices = train_indices[:num_to_use]
            # print(f"Using {args.train_data_percentage}% of data: {len(train_indices)} indices")
        
        # Select examples by indices
        all_train_examples = list(dataset['train'])
        train_examples = [all_train_examples[idx] for idx in train_indices]
        # print(f"Train examples: {len(train_examples)} (from indices file)")
    else:
        # Use index slicing
        train_end_idx = args.train_end_idx if args.train_end_idx is not None else args.num_train_dp
        train_examples = list(dataset['train'])[args.train_start_idx:train_end_idx]
        # print(f"Train examples: {len(train_examples)} (indices {args.train_start_idx}:{train_end_idx})")
    
    val_examples = list(dataset['validation'])
    # print(f"Validation examples: {len(val_examples)}")
    
    # Create datasets
    train_dataset = PromptDataset(train_examples, tokenizer, label_word_list, args.max_length, args.dataset_name)
    val_dataset = PromptDataset(val_examples, tokenizer, label_word_list, args.max_length, args.dataset_name)
    
    # Create generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Create dataloaders with generator for reproducibility
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        generator=g
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    # print("\n[3/6] Creating model...")
    model = PromptBERTForSST2(
        model_name=args.model_name,
        label_word_list=label_word_list,
        num_frozen_layers=args.num_frozen_layers,
        num_labels=2
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer (following FreeShap: Adam with weight_decay)
    # print("\n[4/6] Creating optimizer...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # print(f"Optimizer: Adam(lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Training loop
    # print("\n[5/6] Training...")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.max_epochs + 1):
        # print(f"\n{'='*80}")
        # print(f"Epoch {epoch}/{args.max_epochs}")
        # print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # print(f"\nEpoch {epoch} Results:")
        # print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # # Save best model
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_epoch = epoch
        #     save_path = os.path.join(args.save_dir, 'best_prompt_model.pt')
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_acc': val_acc,
        #         'val_loss': val_loss,
        #     }, save_path)
        #     # print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    # Final evaluation
    # print("\n[6/6] Final Evaluation...")
    # print(f"{'='*80}")
    # print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    # print(f"{'='*80}")
    
    # Load best model and evaluate
    # checkpoint = torch.load(os.path.join(args.save_dir, 'best_prompt_model.pt'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = evaluate(model, val_loader, device)
    # print(f"Final Test Results:")
    # print(f"  Loss: {final_loss:.4f}")
    # print(f"  Accuracy: {final_acc:.4f}")
    # print(f"\nModel saved to: {args.save_dir}/best_prompt_model.pt")
    
    return final_acc


if __name__ == '__main__':
    main()
