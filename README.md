# NeuroBiomark
Univeristy project focused on using DenseNet121 model with attention mechanism techniques as a means for pathology prediction in ALS patients


# Navigation
- GUI: version of the model with a graphical user interface. Can be used to infere images one-by-one.
- Trainable models: contains code for train loops and augmentation evaluation.
-   3 Classes: Original trainable model, contains Concordant, Discordant and Control classes
-   2 Classes: Modified trainable model, contain Concordant and Discordant classes, as well as different augmentation techniques.

# TODO
- Trainable models contain code for EfficientNet and BasicCNN, it clutters the project and makes it hard to understand what functions to what. Better remove them.
- Big chunk of the project is written sequentially by AI, it would be a good idea to tidy up the code into more manageable files, as well as remove unnecessary duplicated functions.
  
