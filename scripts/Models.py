import torch                            # PyTorch to create and apply deep learning models
from torch import nn, optim             # nn for neural network layers and optim for training optimizers
from torch.nn import functional as F    # Module containing several activation functions
import math                             # Useful package for logarithm operations
from data_utils import embedding        # Embeddings and other categorical features handling methods

# [TODO] Create new classes for each model type and add options to include
# variants such as embedding, time decay, regularization learning, etc
class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0, use_batch_norm=True,
                 embed_features=None, num_embeddings=None, embedding_dim=None):
        # [TODO] Add documentation for each model class
        super().__init__()
        self.n_inputs = n_inputs                # Number of input features
        self.n_hidden = n_hidden                # Number of hidden units
        self.n_outputs = n_outputs              # Number of outputs
        self.n_layers = n_layers                # Number of MLP layers
        self.p_dropout = p_dropout              # Probability of dropout
        self.use_batch_norm = use_batch_norm    # Indicates if batch normalization is applied
        self.embed_features = embed_features    # List of features that need to go through embedding layers
        self.num_embeddings = num_embeddings    # List of the total number of unique categories for the embedding layers
        self.embedding_dim = embedding_dim      # List of embedding dimensions
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.num_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of embeddings must also be specified. Received a `embed_features` argument, but not `num_embeddings`.')
            else:
                if isinstance(self.num_embeddings, int):
                    self.num_embeddings = [self.num_embeddings]
                if len(self.num_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `num_embeddings` and the embedding features `embed_features` must have the same length. The provided `num_embeddings` has length {len(self.num_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the current feature;
                        # the formula sets a minimum embedding dimension of 3, with above
                        # values being calculated as the rounded up base 5 logarithm of
                        # the number of embeddings
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.num_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            # Make sure that the embedding_dim is a list
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.num_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # MLP layer(s)
        if self.embed_features is None:
            self.mlp_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as well as the removal
            # of the originating categorical columns
            self.mlp_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        if isinstance(self.n_hidden, int):
            # Make sure that the n_hidden is a list
            self.n_hidden = [self.n_hidden]
        if isinstance(self.n_hidden, list):
            if len(self.n_hidden) != (self.n_layers - 1):
                raise Exception(f'ERROR: The list of hidden units `n_hidden` isn\'t in accordance with the number of layers `n_layers`, which should follow the rule n_hidden = n_layers - 1. The provided `n_hidden` has length {len(self.n_hidden)} while `n_layers` has length {self.n_layers}.')
            # Create a modules list of linear layers
            self.linear_layers = nn.ModuleList()
            # Add the first layer
            self.linear_layers.append(nn.Linear(self.n_inputs, self.n_hidden[0]))
            for i in range(len(self.n_hidden)-1):
                # Add hidden layers
                self.linear_layers.append(nn.Linear(self.n_hidden[i], self.n_hidden[i+1]))
            # Add the last layer, which calculates the final output scores
            self.linear_layers.append(nn.Linear(self.n_hidden[i+1], self.n_outputs))
        else:
            raise Exception(f'ERROR: The `id_columns` argument must be specified as either a single integer or a list of integers. Received input with type {type(n_hidden)}.')
        # Dropout used between the hidden layers
        self.dropout = nn.Dropout(p=self.p_dropout)
        if self.use_batch_norm is True:
            # Batch normalization used between the hidden layers
            self.batch_norm = nn.BatchNorm1d(num_features=self.mlp_n_inputs)

    def forward(self, x):
        if self.embed_features is not None:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                 model_forward=True, inplace=True)
        for i in range(len(self.n_layers)):
            # Apply the current layer
            x = self.linear_layers[i](x)
            if i < self.n_layers - 1:
                # Apply dropout if it's not the final layer
                x = self.dropout(x)
                if self.use_batch_norm is True:
                    # Also apply batch normalization
                    x = self.batch_norm(x)
        # Classification scores after applying all the layers and sigmoid activation
        output = torch.sigmoid(x)
        return output

    def loss(self, y_pred, y_labels):
        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()
        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)
        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()
        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels]
        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred
        return ce_loss
