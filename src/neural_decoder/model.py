import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        use_layer_norm=False,
        input_dropout=0.0,
        use_gradient_checkpointing=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Layer normalization and input dropout for regularization
        self.ln = nn.LayerNorm(neural_dim) if use_layer_norm else nn.Identity()
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        # Protect against sigma=0: use Identity if gaussianSmoothWidth <= 0
        if self.gaussianSmoothWidth <= 0:
            self.gaussianSmoother = nn.Identity()
        else:
            self.gaussianSmoother = GaussianSmoothing(
                neural_dim, 20, self.gaussianSmoothWidth, dim=1
            )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx, lengths=None):
        """
        Forward pass through the GRU decoder.
        
        Args:
            neuralInput: Input tensor of shape [B, T, F]
            dayIdx: Day indices for per-day transformations
            lengths: Optional sequence lengths (T_eff) for pack_padded_sequence.
                    If None, uses full sequence length (no packing).
                    Should be CPU int64 tensor of shape [B]
        
        Returns:
            Output tensor of shape [B, T_out, n_classes+1]
        """
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # Apply layer normalization and input dropout before GRU
        x = transformedNeural
        x = self.ln(x)
        x = self.in_drop(x)
        
        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # Speed optimization: Use pack_padded_sequence to skip padded tokens
        # This speeds up GRU by avoiding computation on padding and saves memory
        # NOTE: pack_padded_sequence is incompatible with torch.compile (requires CPU tensors)
        # We disable packing only when actively being compiled (during torch.compile tracing)
        use_packing = lengths is not None
        if use_packing and hasattr(torch, '_dynamo'):
            # Only disable packing during the actual compilation phase
            # After compilation, we can use packing if torch.compile is disabled
            try:
                if torch._dynamo.is_compiling():
                    use_packing = False
            except (AttributeError, RuntimeError, TypeError):
                # If check fails, assume we're not compiling and allow packing
                pass
        
        # Use gradient checkpointing if enabled (trades compute for memory)
        if self.use_gradient_checkpointing:
            # Gradient checkpointing: recompute activations during backward pass
            # This saves memory at the cost of extra forward passes
            if use_packing:
                # lengths should be CPU int64 for pack_padded_sequence
                lengths_cpu = lengths.cpu().to(torch.int64)
                # Pack sequences (sorts by length internally for efficiency)
                packed = pack_padded_sequence(
                    stridedInputs, 
                    lengths=lengths_cpu, 
                    batch_first=True, 
                    enforce_sorted=False
                )
                # Use checkpointing for GRU forward pass
                def gru_forward(packed_input):
                    return self.gru_decoder(packed_input)
                hid, _ = checkpoint(gru_forward, packed, use_reentrant=False)
                # Unpack back to padded format
                hid, _ = pad_packed_sequence(hid, batch_first=True)
            else:
                # Checkpointing for non-packed sequences
                def gru_forward(input_seq):
                    return self.gru_decoder(input_seq)
                hid, _ = checkpoint(gru_forward, stridedInputs, use_reentrant=False)
        elif use_packing:
            # lengths should be CPU int64 for pack_padded_sequence
            lengths_cpu = lengths.cpu().to(torch.int64)
            # Pack sequences (sorts by length internally for efficiency)
            packed = pack_padded_sequence(
                stridedInputs, 
                lengths=lengths_cpu, 
                batch_first=True, 
                enforce_sorted=False
            )
            hid, _ = self.gru_decoder(packed)
            # Unpack back to padded format
            hid, _ = pad_packed_sequence(hid, batch_first=True)
        else:
            # Fallback: no packing (use full sequence)
            # This is used when torch.compile is active or lengths not provided
            hid, _ = self.gru_decoder(stridedInputs)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
