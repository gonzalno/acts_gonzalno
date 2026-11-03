import torch 

class MLP(torch.nn.Module):
    """
    Class for constructing MLPs taking 
    hyperparameters as configuration 
    options.
    """
    def __init__(self, input_dims:int, output_dims:int, activation:torch.nn, nlayers:int, layer_dims:int):
        super().__init__()
        self.indim = input_dims
        self.outdim = output_dims
        self.activation = activation
        #Build a list of nn components
        nn = [torch.nn.Linear(input_dims, layer_dims), self.activation]
        #Loop through the number of layers
        for layer in range(nlayers):
            nn.append(torch.nn.Linear(layer_dims, layer_dims))
            nn.append(self.activation)
        #Add the output layer
        nn.append(torch.nn.Linear(layer_dims, output_dims))
        #Unpack the network
        self.mlp = torch.nn.Sequential(*nn)
    def forward(self, x:torch.tensor)->torch.tensor:
        """
        Forward method for mlp, returns torch 
        tensor.
        """
        return self.mlp(x)


class gate(torch.nn.Module):
    """
    Builds a Gate function for Track <-> hit association which is 
    decomposed into a reusble embedding network and the classifier head.
    """
    def __init__(self, embedding_nn:torch.nn.Module, classifier_nn:torch.nn.Module):
        super().__init__()
        self.embed = embedding_nn
        self.classifier = classifier_nn
    def forward(self, x:torch.tensor)->tuple:
        """
        Runs the forward method for the embedding nn
        and classifier to produce a tuple output of the
        form (embedded x, output logit).
        """
        embedded_x = self.embed.forward(x)
        out_logit = self.classifier.forward(embedded_x)
        return embedded_x, out_logit