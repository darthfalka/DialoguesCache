
# Nostalgic footprints

This respiratory contains experiments on modeling problems using word tokens. Rather than relying on popular models like transformers, I decided to revisit some earlier NLP models from the past.

##### Main Objective

Constructing an entity-relational graph using speech embeddings from pretrained models. The goal is to structure relationships between entities by applying distance-based loss functions to minimise/maximise the distance between wanted/unwanted pairings, respectively.

This approach is valuable for building speech-driven knowledge graphs and  entity linking systems. By hooking a network layer on top of embeddings generated from pre-trained models, the experiment aims to transform the embeddings based on the most recent observations. This can be used for decision-making and reasoning in tasks like chatbots or AI assistants.

##### Other objectives

* Further analysis in word / token embeddings.
* More experiments with crafting networks from scratch.

##### Data

The data used in this repository includes pre-trained GloVe embeddings:

- https://nlp.stanford.edu/projects/glove/
