# Protest event detection
This code can be used to run protest event detection models that were trained as part of my master thesis. There are four different models:
- **Haystack classifier**: Predict whether a given text is about a protest (binary classification).
- **Haystack and *form* classifier**: Predict whether a given text is about a protest, and the form of the protest (multiclass classification).
- **Haystack and *issue* classifier**: Predict whether a given text is about a protest, and the issue of the protest (multiclass classification).
- **Haystack and *target* classifier**: Predict whether a given text is about a protest, and the target of the protest (multiclass classification).
- **Full multitask classifier**: Jointly predicts all the above tasks.

##### List of classes
<details>
  <summary>Haystack classes</summary>

  1. Non-protest
  2. Protest

</details>
<details>
  <summary>Form classes</summary>

  1. Blockade/slowdown/disruption
  2. Boycott
  3. Hunger strike
  4. March
  5. Non-protest
  6. Rally/demonstration
  7. Riot
  8. Strike/walkout/lockout

</details>
<details>
  <summary>Issue classes</summary>

  1. Anti-colonial/political independence
  2. Anti-war/peace
  3. Criminal justice system
  4. Democratisation
  5. Economy/inequality
  6. Environmental
  7. Foreign policy
  8. Human and civil rights
  9. Labour & work
  10. Non-protest
  11. Political corruption/malfeasance
  12. Racial/ethnic rights
  13. Religion
  14. Social services & welfare
  15. None of the above

</details>
<details>
  <summary>Target classes</summary>

  1. Domestic government
  2. Foreign government
  3. Individual
  4. Intergovernmental organisation
  5. Non-protest
  6. Private/business

</details>

### How-to
Running the Python (3.7+) script requires a couple of packages:
- [PyTorch 1.4](https://pytorch.org/get-started/locally/) (works with and without CUDA)
- [huggingface transformers](https://github.com/huggingface/transformers)
- [tqdm](https://pypi.org/project/tqdm/)
- [numpy](https://numpy.org/)

The models can be downloaded from [here](https://www.dropbox.com/s/61cqlvharan4xkz/models.tar.gz?dl=0) (approx. 4.2GB). The file must be extracted into the `src` directory, and this can be done using e.g. [7-zip](https://www.7-zip.org/) on Windows or with [`tar`](https://www.cyberciti.biz/faq/how-to-create-tar-gz-file-in-linux-using-command-line/) on Linux/Mac.

Using `tar`, the command is
```
tar -xzvf models.tar.gz -C path/to/protest-event-detection/src/
```

The main script to run is `classify_article.py`. It has several parameters:
- `--article`: Path to a text file with the raw article text to make predictions on. Optimal article length is around 350 words (maximum).
- `--output_path` (optional): Path to where the prediction file will be stored. At default the script does not output predictions to file, but to terminal.
- `--out_file` (optional): Name of the output file where the prediction will be stored, if the above parameter is set. Default name: `pred.txt`.
- `--task`: Which prediction model to use. Possible values are haystack, form, issue, target and multi.
- `--mc_samples` (optiona): Number of times to make predictions on the same article. This sets the model in training mode, such that its predictions are stochastic. Then, the final prediction is based on an average of the number of predictions made. This gives additional output (uncertainty estimates). If not set, the model makes a single prediction in evaluation mode. Recommended value is 50 if not zero. Default value: 0.
- `--gpu_devices` (optional): If CUDA is available, will use GPU(s). Here, one can specify a GPU to use, e.g. `--gpu_devices 1` if there are multiple GPUs in the system and you want to use the second GPU. With CUDA available, using a single GPU is more than enough to classify one article. Normally, it is not necessary to modify this option. Default value: 0 (the first GPU).

**Note**: Depending on your setup, running predictions without CUDA/GPU is (relatively) very slow.

### Run examples

Make a single haystack prediction without uncertainty estimates:
```
python classify_article.py --article ../path/to/article.txt --task haystack
```

Make 50 Monte Carlo haystack predictions and output to file:
```
python classify_article.py --article ../path/to/article.txt --mc_samples 50 --task haystack --output_path ../some/path/
```

Make 50 Monte Carlo haystack and *form* predictions and output to file with specific name:
```
python classify_article.py --article ../path/to/article.txt --mc_samples 50 --task form --output_path ../some/path/ --out_file prediction.txt
```

Make 50 Monte Carlo predictions for **all** tasks and output to file:
```
python classify_article.py --article ../path/to/article.txt --mc_samples 50 --task multi --output_path ../some/path/
```
