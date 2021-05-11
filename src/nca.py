from torch import nn
import torch

class NCA(nn.Module):
    """Our custom Nerual Collaborative Filtering Model.

        Parameters
        ----------
        config : dictionary
            Settings of the model.

        Attributes
        ----------
        n_users : np.int
            Maximum number of users in the system.
        n_items : np.int
            Maximum number of items (e.g. movies, products...) in the system.
        k : np.int
            Factorization rank.
        embed_user : torch.nn.Embedding
            Embedding layer for users data.
        embed_item : torch.nn.Embedding
            Embedding layer for users data.
        fc_layers : torch.nn.ModuleList
            List of forward fully connected linear layers.
        dropout : torch.nn.Dropout
            Dropout layer.
        output : torch.nn.Linear
            The output layer of the network.
        output_f : torch.nn.functional.Sigmoid
            The activation function of the output layer.
        config : dictionary
            The training configurations passed in the constructor of NCA class.

    """


    def __init__(self, config):
        """constructor.

        Parameters
        ----------
        config : dictionary
            The settings of the neural network.

        Returns
        -------
        None

        """
        super().__init__()

        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.k = config['k']

        # Define the layers
        self.embed_user = nn.Embedding(self.n_users, self.k)
        self.embed_item = nn.Embedding(self.n_items, self.k)

        # The stacked Linear layers in which the in_features and out_features are managed using this loop
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))


        self.dropout = nn.Dropout(0.2)

        # The output layer
        self.output = nn.Linear(config['layers'][-1],  1)

        # The activation function of the last layer
        self.output_f = nn.Sigmoid()

    def forward(self, users, items):
        """Forward pass on the data.

        Parameters
        ----------
        users : torch.Tensor
            Users tensor.
        items : torch.Tensor
            Items tensor.

        Returns
        -------
        torch.Tesnor
            The tensor after the forward pass.

        """

        # Embedding users and items separately
        users_x = self.embed_user(users)
        items_x = self.embed_item(items)

        # Concatenate them
        x = torch.cat([users_x, items_x], dim = 1) # Concatenate along the second axis

        # The in_features for the first fc layer will be determined according to the preprocessing step
        n = 2 # Users  + Items
        if self.config['one_hot_encoding']:
            n = (self.n_users + self.n_items ) # One hot encoding
        else:
            n = 2 # no one hot encoding

        # Reshape the input to the fc layers
        x = x.view(-1, n * self.k)
    

        for i in range(len(self.fc_layers)):
          x = self.fc_layers[i](x)
          x = nn.ReLU()(x)
          x = self.dropout(x)

        x = self.output(x)

        # Scaling the output from [0-1] to [1-5]
        x = self.output_f(x) * self.config['rating_range'] + self.config['lowest_rating']
        return x
