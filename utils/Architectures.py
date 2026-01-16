import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat
from vit_pytorch import ViT
from utils.Architectures_utils import *
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        # LOS-based
        'LOS-Net': LOS_Net,
        'ImprovedLOSNet': ImprovedLOSNet,
        'ATP_R_MLP': ATP_R_MLP,
        'ATP_R_Transf': ATP_R_Transf,
    }

    if args.probe_model in {'LOS-Net', 'ImprovedLOSNet', 'ATP_R_Transf'}:
        return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
    elif args.probe_model in {'ATP_R_MLP'}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)
    else:
        raise ValueError(f"Unknown model: {args.probe_model}")
    

######################## LOS ########################
class ATP_R_MLP(nn.Module):

    def __init__(self, args, actual_sequence_length):

        super(ATP_R_MLP, self).__init__()        
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.actual_sequence_length = actual_sequence_length
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")

        
        # Linear layers
        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim * self.actual_sequence_length
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Output act
        self.sigmoid = nn.Sigmoid()
    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):


        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        x = encoded_ATP_R + encoded_normalized_ATP
        x = x.flatten(start_dim=1)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)


        return self.sigmoid(x).squeeze(-1)  # Apply sigmoid for binary classification


class ATP_R_Transf(nn.Module):
    
    def __init__(self, args, max_sequence_length, input_dim=1):
        
        super(ATP_R_Transf, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
        
        

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Positional embeddings with a predefined max sequence length
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # Classification head
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
            
        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        x = encoded_ATP_R + encoded_normalized_ATP

    
        # Add [CLS] token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # Shape: [B, 1, hidden_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, hidden_dim]

        # Generate positional indices and add embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)  # Shape: [1, N+1]
        pos_embeddings = self.pos_embedding(pos_indices)  # Shape: [1, N+1, hidden_dim]
        x += pos_embeddings

        # Pass through Transformer layers
        for layer in self.attention_layers:
            x = layer(x)  # Shape remains [B, N+1, hidden_dim]

        # Pooling: Use the CLS token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # Final classification head
        x = self.mlp_head(x)  # Shape: [B, 1]
        
        return self.sigmoid(x).squeeze(-1)  # Apply sigmoid for binary classification
    

class LOS_Net(nn.Module):
    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        
        assert self.pool in {'cls', 'mean'}, "Pool type must be either 'cls' (CLS token) or 'mean' (mean pooling)"
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))


        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim // 2,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
        
        
        
        # Input embedding layer
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        
        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Classification head
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        """
        Forward pass for LOS_Net.

        Args:
            sorted_TDS_normalized (torch.Tensor): Shape [B, N, V].
            normalized_ATP (torch.Tensor): Shape [B, N, 1].
            ATP_R (torch.Tensor): Shape [B, N].
            sigmoid (bool): Whether to apply sigmoid activation. Default is True.

        Returns:
            torch.Tensor: Output tensor of shape [B, 1] (if sigmoid=True) or raw logits (if sigmoid=False).
        """
        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
            
        
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        
        # Encoding normalized vocab
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))
        
        # Concatenating embeddings
        x = torch.cat((encoded_sorted_TDS_normalized, encoded_ATP_R + encoded_normalized_ATP), dim=-1)
        
        # Adding CLS token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)
        
        # Transformer layers
        for layer in self.attention_layers:
            x = layer(x)
        
        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # Classification head
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)


class ImprovedLOSNet(nn.Module):
    """
    Enhanced LOS-Net with improved feature encoding:
    1. Learned rank embeddings (instead of simple scaling)
    2. Entropy features (distribution uncertainty)
    3. Probability gap features (confidence measures)
    """
    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()

        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool

        # Feature flags
        self.use_rank_embed = args.use_rank_embed
        self.use_entropy = args.use_entropy
        self.use_gaps = args.use_gaps

        assert self.pool in {'cls', 'mean'}, "Pool type must be either 'cls' (CLS token) or 'mean' (mean pooling)"

        # Original ATP encoding parameter
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))

        # NEW: Learned rank embeddings
        if self.use_rank_embed:
            self.rank_embedding = nn.Embedding(
                num_embeddings=args.topk_dim,  # K tokens
                embedding_dim=args.rank_embed_dim
            )

        # Input embedding layer for TDS
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)

        # Compute combined feature dimension
        combined_dim = self.hidden_dim // 2  # TDS features
        combined_dim += self.hidden_dim // 2  # ATP features
        if self.use_rank_embed:
            combined_dim += args.rank_embed_dim
        if self.use_entropy:
            combined_dim += 1
        if self.use_gaps:
            combined_dim += 2

        # Project combined features to hidden dimension
        self.feature_projection = nn.Linear(combined_dim, self.hidden_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Positional embeddings
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # Classification head
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_entropy(self, sorted_TDS_normalized):
        """
        Compute Shannon entropy of probability distributions.

        Args:
            sorted_TDS_normalized: [B, N, K] - normalized sorted probabilities

        Returns:
            entropy: [B, N, 1] - entropy for each position
        """
        # Convert normalized values back to probabilities
        # Note: sorted_TDS_normalized is z-score normalized, need to convert to probs
        # For now, use softmax to ensure valid probability distribution
        probs = F.softmax(sorted_TDS_normalized, dim=-1)

        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1, keepdim=True)

        return entropy

    def compute_prob_gaps(self, sorted_TDS_normalized):
        """
        Compute gaps between top probabilities.

        Args:
            sorted_TDS_normalized: [B, N, K] - normalized sorted probabilities

        Returns:
            gaps: [B, N, 2] - [gap between top-1 and top-2, gap between top-1 and top-10]
        """
        # Gap 1: difference between highest and second-highest probability
        gap1 = sorted_TDS_normalized[..., 0:1] - sorted_TDS_normalized[..., 1:2]

        # Gap 2: difference between highest and 10th-highest probability
        gap2 = sorted_TDS_normalized[..., 0:1] - sorted_TDS_normalized[..., 9:10]

        gaps = torch.cat([gap1, gap2], dim=-1)
        return gaps

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        """
        Forward pass for ImprovedLOSNet.

        Args:
            sorted_TDS_normalized: [B, N, K] - sorted token probabilities
            normalized_ATP: [B, N, 1] - actual token probability
            ATP_R: [B, N] - rank of actual token

        Returns:
            predictions: [B] - binary predictions
        """
        B, N, K = sorted_TDS_normalized.shape

        # === STEP 1: Encode TDS (same as baseline) ===
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))  # [B, N, hidden_dim//2]

        # === STEP 2: Encode ATP (same as baseline) ===
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP  # [B, N, hidden_dim//2]

        # === STEP 3: NEW - Learned Rank Embeddings ===
        feature_list = [encoded_sorted_TDS_normalized, encoded_normalized_ATP]

        if self.use_rank_embed:
            # Create rank indices [0, 1, 2, ..., K-1] for each position
            rank_indices = torch.arange(K, device=sorted_TDS_normalized.device)
            rank_indices = rank_indices.unsqueeze(0).unsqueeze(0).expand(B, N, K)  # [B, N, K]

            # Get rank embeddings
            rank_embeds = self.rank_embedding(rank_indices)  # [B, N, K, rank_embed_dim]

            # Aggregate across K dimension (mean pooling)
            rank_embeds = rank_embeds.mean(dim=2)  # [B, N, rank_embed_dim]

            feature_list.append(rank_embeds)

        # === STEP 4: NEW - Entropy Features ===
        if self.use_entropy:
            entropy = self.compute_entropy(sorted_TDS_normalized)  # [B, N, 1]
            feature_list.append(entropy)

        # === STEP 5: NEW - Probability Gap Features ===
        if self.use_gaps:
            gaps = self.compute_prob_gaps(sorted_TDS_normalized)  # [B, N, 2]
            feature_list.append(gaps)

        # === STEP 6: Concatenate All Features ===
        x = torch.cat(feature_list, dim=-1)  # [B, N, combined_dim]

        # === STEP 7: Project to Hidden Dimension ===
        x = self.feature_projection(x)  # [B, N, hidden_dim]

        # === STEP 8: Transformer Processing (same as baseline) ===
        # Adding CLS token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, hidden_dim]

        # Positional embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)

        # Transformer layers
        for layer in self.attention_layers:
            x = layer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # Classification head
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)

######################## LOS ########################
