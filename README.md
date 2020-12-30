# sam.pytorch

A PyTorch implementation of *Sharpness-Aware Minimization for Efficiently Improving Generalization* (
Foret+2020) [Paper](https://arxiv.org/abs/2010.01412), [Official implementation](https://github.com/google-research/sam)
.

## Requirements

* Python>=3.8
* PyTorch>=1.7.1

To run the example, you further need

* `homura` by `pip install -U homura-core==2020.12.0`
* `chika` by `pip install -U chika`

## Example

```commandline
python cifar10.py [--optim.name {sam,sgd}] [--model {renst20, wrn28_2}] [--optim.rho 0.05]
```

### Results: Test Accuracy (CIFAR-10)

Model       | SAM | SGD |
---         | --- | --- |
ResNet-20   | 93.5| 93.2|
WRN28-2     | 95.8| 95.4|

SAM needs double forward passes per each update, thus training with SAM is slower than training with SGD. In case of
ResNet-20 training, 80 mins vs 50 mins on my environment. Additional options `--use_amp --jit_model` may slightly
accelerates the training.

## Usage

`SAMSGD` can be used as a drop-in replacement of PyTorch optimizers with closures. Also, it is compatible
with `lr_scheduler` and has `state_dict` and `load_state_dict`.

```python
from sam import SAMSGD

optimizer = SAMSGD(model.parameters(), lr=1e-1, rho=0.05)

for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_f(output, target)
        loss.backward()
        return loss


    loss = optimizer.step(closure)
```

## Citation

```bibtex
@ARTICLE{2020arXiv201001412F,
    author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
    title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
    year = 2020,
    eid = {arXiv:2010.01412},
    eprint = {2010.01412},
}

@software{sampytorch
    author = {Ryuichiro Hataya},
    titile = {sam.pytorch},
    url    = {https://github.com/moskomule/sam.pytorch},
    year   = {2020}
}
```