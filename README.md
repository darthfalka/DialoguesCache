
# Furry Tokens

This respiratory contains experiments on modeling problems using word tokens. Rather than relying on popular models like transformers, I decided to revisit some earlier NLP models from the past.

##### Main Objective

* Constructing an entity-relational graph using speech embeddings from pretrained models (e.g., Wav2Vec2). The goal is to structure relationships between entities by applying distance-based loss functions to minimise/maximise the distance between wanted/unwanted pairings, respectively. The approach is useful for building speech-driven knowledge graphs and entity linking systems via network layer that is able to be hooked ontop of embeddings that was generated from pre-trained models. This experimenting layer's should be able to transform the embeddings to the most recent observation. This can be used as the expected output for a bot's decision making and reasoning process.

##### Other objectives

* Further analysis in word / token embeddings.
* More experiments with crafting networks from scratch.
