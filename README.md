This Github Repository contains all code and generations to reproduce results found in the paper:

[Predict the Next Word: \<Humans exhibit uncertainty in this task and language models _____\>](https://arxiv.org/abs/2402.17527)

The raw data used, obtained from [Provo Corpus](https://link.springer.com/article/10.3758/s13428-017-0908-4), can be found in the `raw_data` folder.

Within each model's folder (in the `TVD` folder) you will find a script allowing for generating samples from the model, the generations we used for our analysis (in the sub-folder `Generations`), a script processing and analysing the generations and the uncertainty estimates produced from the script (in the sub-folder `uncertainty_results`). A notebook visualising jointly the results from all models can be found in the same folder, under the name `joint_uncertainty_analysis.ipynb`.

For the experiments of the paper regarding syntax- and semantic-level experiments, as described in the *Results* section of the paper, you can consult `TVD_semantic` and `TVD_syntactic` folders.

Lastly, to reproduce experiments and figures of the *Appendix*, consult the `Additional_Experiments` folder, organised by Appendix sections.



**Acknowledgement**:
This work was funded by the European Union's Horizon Europe (HE) Research and Innovation programme under Grant Agreement No 101070631 and from the UK Research and Innovation (UKRI) under the UK government's HE funding grant No 10039436.
