# Biological_Learning
Example of "biological" learning for MNIST based on the paper [Unsupervised Learning by Competing Hidden Units](https://doi.org/10.1073/pnas.1820458116) by D.Krotov and J.Hopfield. If you want to learn more about this work you can also check out this [lecture](https://www.youtube.com/watch?v=4lY-oAY0aQU) from MIT's [6.S191 course](http://introtodeeplearning.com/).  

## Getting started

install jupyter notebook and numpy, scipy, matplotlib.

```bash
> jupyter notebook
```
run `Unsupervised_learning_algorithm_MNIST.ipynb` and observe weights.

## Author and License
(c) 2018 Dmitry Krotov
-- Apache 2.0 License

### Original MLP
Epoch:  1       Training Loss: 0.886063         Validation Loss: 0.106482
Validation loss decreased (inf --> 0.106482). Saving model...
Epoch:  2       Training Loss: 0.355428         Validation Loss: 0.077120
Validation loss decreased (0.106482 --> 0.077120). Saving model...
Epoch:  3       Training Loss: 0.280633         Validation Loss: 0.065516
Validation loss decreased (0.077120 --> 0.065516). Saving model...
Epoch:  4       Training Loss: 0.238252         Validation Loss: 0.057179
Validation loss decreased (0.065516 --> 0.057179). Saving model...
Epoch:  5       Training Loss: 0.206558         Validation Loss: 0.050144
Validation loss decreased (0.057179 --> 0.050144). Saving model...
Test Loss: 0.180058

Test Accuracy of     0: 98% (968/980)
Test Accuracy of     1: 98% (1114/1135)
Test Accuracy of     2: 94% (974/1032)
Test Accuracy of     3: 93% (944/1010)
Test Accuracy of     4: 93% (922/982)
Test Accuracy of     5: 92% (824/892)
Test Accuracy of     6: 95% (918/958)
Test Accuracy of     7: 92% (955/1028)
Test Accuracy of     8: 93% (907/974)
Test Accuracy of     9: 92% (937/1009)

Test Accuracy (Overall): 94% (9463/10000)
### Hebbian fc1 weight MLP
Epoch:  1       Training Loss: 1.118378         Validation Loss: 0.240063
Validation loss decreased (inf --> 0.240063). Saving model...
Epoch:  2       Training Loss: 0.938317         Validation Loss: 0.220884
Validation loss decreased (0.240063 --> 0.220884). Saving model...
Epoch:  3       Training Loss: 0.878558         Validation Loss: 0.214125
Validation loss decreased (0.220884 --> 0.214125). Saving model...
Epoch:  4       Training Loss: 0.833336         Validation Loss: 0.204880
Validation loss decreased (0.214125 --> 0.204880). Saving model...
Epoch:  5       Training Loss: 0.804351         Validation Loss: 0.192989
Validation loss decreased (0.204880 --> 0.192989). Saving model...
Test Loss: 0.571776

Test Accuracy of     0: 95% (931/980)
Test Accuracy of     1: 98% (1122/1135)
Test Accuracy of     2: 91% (946/1032)
Test Accuracy of     3: 73% (738/1010)
Test Accuracy of     4: 86% (851/982)
Test Accuracy of     5: 84% (751/892)
Test Accuracy of     6: 93% (894/958)
Test Accuracy of     7: 86% (891/1028)
Test Accuracy of     8: 82% (799/974)
Test Accuracy of     9: 64% (654/1009)

Test Accuracy (Overall): 85% (8577/10000)