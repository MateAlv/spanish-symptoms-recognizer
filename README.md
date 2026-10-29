> **Esta es la copia que usa [diagnostic.ar](https://github.com/MateAlv/diagnosticar-plataforma).** El desarrollo original del reconocedor, con sus experimentos, está en [GMBUST-FIUBA/spanish-symptoms-recognizer](https://github.com/GMBUST-FIUBA/spanish-symptoms-recognizer). Acá se suman los cambios para integrarlo a la plataforma: elegir el proveedor de LLM, responder en el formato de `/extract-hpo` y publicar la imagen.

# Sypmtoms recognizer in spanish

## Introduction
This work is made as part of a final project in the University of Buenos Aires, Argentina, named *Detection of Genetic and Rare Deseases* (*Detección de enfermedades genéticas poco frecuentes*, in spanish). This repository contains all code used for training the models used both for detecting symptoms and mapping symptoms.

Based on the model developed by the Barcelona Supercomputing Center (BSC) named [bsc-bio-ehr-es](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es), a fine-tuned version of said model is made in order to recognize symptoms in clynical history. Those recognized symptoms can then be mapped on to an ontology, however currently the only ontology used is [HPO](https://hpo.jax.org/) since it is the one that was decided to use as the standard in the development of this project.

Both organic and synthetic datasets where used to train the models to use the [HPO](https://hpo.jax.org/), as will be explained later.

## Training

### Organic data
The model was trained using the following datasets:

- [SympTEMIST Corpus](https://zenodo.org/records/10635215): This dataset contains a very complete set of annotations for NER models.
- IN PROGRESS: [Chilean Waiting List](https://zenodo.org/records/7555181)
- IN THE FUTURE: [eHealth-KD v2](https://zenodo.org/records/3696792)
- IN THE FUTURE: [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus)

### Synthetic data
To ensure that the model can detect all HPO phenotypes a synthetic dataset was developed to generate cases where at least all phenotypes are observed and detected.


## Mapping
As stated before, the only ontology that it it supported is [HPO](https://hpo.jax.org/). The translations used are those made by the 
Human Phenotype Ontology Internationalisation Effort's [spanish translation](https://github.com/obophenotype/hpo-translations/blob/main/babelon/hp-es.babelon.tsv).