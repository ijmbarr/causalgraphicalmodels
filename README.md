# CausalGraphicalModels

## Introduction

`causalgraphicalmodels` is a python module for describing and manipulating [Causal Graphical Models](https://en.wikipedia.org/wiki/Causal_graph) and [Structural Causal Models](https://en.wikipedia.org/wiki/Structural_equation_modeling). Behind the scenes it is a light wrapper around the python graph library [networkx](https://networkx.github.io/), together with some CGM specific tools.

It is currently in a very early stage of development. All feedback is welcome.


## Example

For a quick overview of `CausalGraphicalModel`, see [this example notebook](https://github.com/ijmbarr/causalgraphicalmodels/blob/master/notebooks/cgm-examples.ipynb).

## Install

```
pip install causalgraphicalmodels
```


## Resources
My understanding of Causality comes mainly from the reading of the follow work:
 - Causality, Pearl, 2009, 2nd Editing. (An overview available [here](http://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf))
 - A fantastic blog post, [If correlation doesn’t imply causation, then what does?](http://www.michaelnielsen.org/ddi/if-correlation-doesnt-imply-causation-then-what-does/) from Michael Nielsen
 - [These lecture notes](http://www.math.ku.dk/~peters/jonas_files/scriptChapter1-4.pdf) from Jonas Peters
 - The draft of [Elements of Causal Inference](http://www.math.ku.dk/~peters/jonas_files/bookDRAFT5-online-2017-02-27.pdf)
 - http://mlss.tuebingen.mpg.de/2017/speaker_slides/Causality.pdf

## Related Packages
 - [Causality](https://github.com/akelleh/causality)
 - [CausalInference](https://github.com/laurencium/causalinference)
 - [DoWhy](https://github.com/Microsoft/dowhy)



