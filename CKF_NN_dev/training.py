import torch
import numpy as np 
import awkward as ak
from glob import glob
from gate_network import (MLP, gate)
import matplotlib.pyplot as plt
import hist 
import mplhep as hep
hep.style.use("ATLAS")

label_files = glob("../ColliderML_processed/torch_data/ttbar_hard_scatter/*labels.pt")

feature_files = glob("../ColliderML_processed/torch_data/ttbar_hard_scatter/*features.pt")

#Create Gate Neural Network

#Two options for classifier head

#FFN with SiLU
embed_nn = MLP(12, 256, torch.nn.SiLU(), 2, 256)
classifier_head1 = MLP(256, 1, torch.nn.SiLU(), 2, 256)
Gate1 = gate(embed_nn, classifier_head1)

#Classifier is GLU
classifier_head2 = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                        torch.nn.SiLU(),
                                        torch.nn.Linear(256, 256),
                                        torch.nn.GLU(),
                                        torch.nn.Linear(128, 1)
                                        )

Gate2 = gate(embed_nn, classifier_head2)

#Optimizer
optimizer = torch.optim.AdamW(Gate1.parameters(), lr = 1e-4, weight_decay = 1e-6)

#Classifier head models Bernoulli 
BCE = torch.nn.BCEWithLogitsLoss(reduction="none")

#Positive and negative counting for 
#Prior term in Loss
npos = 0
ntot = 0
ntrain = 0


testing_set = []
validation_set = []
training_set = []

train_split = 0.6
val_split = 0.2
test_split = 0.2

nepochs = 10

for label_file in label_files:
    event_num = label_file.split("/")[-1].split("_")[2]
    feature_file = [f for f in feature_files if event_num==f.split("/")[-1].split("_")[2]][0]
    #Load the features
    fmodule = torch.load(feature_file, weights_only=False)
    try:
        #Reshape with stack to EVENTS x FEATURES
        features = torch.stack( [param for _, param in fmodule.named_parameters() ] )
    except:
        print("EMPTY FILE SKIPPING")
        continue
    #Load the labels
    lmodule = torch.load(label_file, weights_only=False)
    #Reshape with stack to EVENTS x 1
    labels = torch.stack( [param for _, param in lmodule.named_parameters()] )
    #Create a torch Dataset 
    Dataset = torch.utils.data.TensorDataset(features, labels)
    #Split 
    train_events, val_events = int(len(labels)*train_split), int(len(labels)*val_split)
    ntrain += train_events
    train_part_set, val_part_set, test_part_set = torch.utils.data.random_split(Dataset, [train_events, val_events, len(labels) - train_events - val_events])
    training_set.append(train_part_set)
    validation_set.append(val_part_set)
    testing_set.append(test_part_set)
    #Keep track of the ratio positive to negative examples 
    #for the prior 
    npos += torch.sum(labels.squeeze()).item()
    ntot += len(labels)

print(npos, ntot)

#Prior term in the NLL for Bernoulli 
prior = np.log(npos/(ntot-npos))

#Scheduler set after we know the total size of the training set
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ntrain//128 + 1, eta_min = 1e-6)

#Create the large datasets
training = torch.utils.data.DataLoader( torch.utils.data.ConcatDataset(training_set), shuffle=True, batch_size = 128 )
validation = torch.utils.data.DataLoader( torch.utils.data.ConcatDataset(validation_set), shuffle=False, batch_size = 512 )
testing = torch.utils.data.DataLoader( torch.utils.data.ConcatDataset(testing_set), shuffle=False, batch_size = 512 )

for epoch in range( nepochs ):
    print(f"Starting Epoch {epoch}")
    for batch in training:
        optimizer.zero_grad()
        fs = batch[0]
        #Squeeze the labels to be EVENTS
        ls = batch[1].squeeze()
        #Run the forward pass
        embedded, out_logit = Gate1.forward(fs)
        loss = BCE(out_logit.squeeze(), ls.float()) - prior
        loss = torch.mean( loss )
        loss.backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        val_loss = []
        val_size = 0
        for batch in validation:
            fs = batch[0]
            #Squeeze the labels to be EVENTS
            ls = batch[1].squeeze()
            val_size += ls.size()[0]
            #Run the forward pass
            embedded, out_logit = Gate1.forward(fs)
            loss = BCE(out_logit.squeeze(), ls.float()) - prior
            val_loss.append( torch.sum(loss) )
        val_losses = torch.sum( torch.tensor( val_loss ) ) / val_size
        print( val_losses )


test_out_logits = []
test_labels = []

with torch.no_grad():
    for batch in testing:
        fs = batch[0]
        #Squeeze the labels to be EVENTS
        ls = batch[1].squeeze()
        val_size += ls.size()[0]
        #Run the forward pass
        embedded, out_logit = Gate1.forward(fs)
        test_labels.append(ls.cpu().numpy())
        test_out_logits.append(out_logit.cpu().numpy())

test_labels = ak.flatten(test_labels)
test_out_logits = 1 / ( 1 + np.exp( - ak.flatten( ak.flatten(test_out_logits) ) ) )


test_evaluation = hist.Hist(hist.axis.Regular(20, 0, 1, name="score", label="Score"),
                            hist.axis.StrCategory(["Positive Examples", "Negative Examples"], name="dataset"))

test_evaluation.fill(score = test_out_logits[test_labels==1], dataset = "Positive Examples")

test_evaluation.fill(score = test_out_logits[test_labels==0], dataset = "Negative Examples")

test_evaluation.plot()

plt.yscale("log")

plt.savefig("test_scores_ML_CKF_GATE.png")