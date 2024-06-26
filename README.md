# TransferLab Training: Simulation-Based Inference

Welcome to the TransferLab Training: Simulation-Based Inference (SBI). This is
the readme for participants of the training.

Simulation-based inference is a statistical method used to enable Bayesian
inference when the likelihood function of a complex model is computationally
intractable or unknown. It achieves this by using neural network-based density
estimation methods to approximate a specific part of Bayes' theorem using data
simulated from the model.

This training provides participants with a thorough understanding of the
fundamental principles and methods of SBI, including when and why to use these
methods instead of traditional likelihood-based inference techniques. The
training also offers hands-on experience in applying SBI.

<figure>
    <center>
    <img src="./notebooks/_static/images/sbi_concept_figure.png" style="width:100%"/>
    <figcaption>Schematic overview of the Simulation-based Inference workflow (Jan Boelts, 2023).</figcaption>
</figure>

## Training Agenda

You can find the agenda for the training in the file [`AGENDA`](./AGENDA.md).
Everything is already set up, so you can either follow the trainer's
presentation or explore the notebooks and source code on your own.

## Running the Notebooks

Participants will have access to a Jupyter Hub during the training to run the
notebooks. However, you might also want to pull the repository and execute it
locally. Below are the steps for running the content:

### Viewing the Notebooks

1. To simply view the rendered notebooks, open `html/index.html` in your
   browser.

### Executing the Notebooks Locally

#### Without Docker

1. Create a conda environment with Python 3.11:

    ```shell
    conda create -n tfl_training_sbi python=3.11
    ```

2. Install the dependencies and the package:

    ```shell
    bash build_scripts/install_presentation_requirements.sh
    pip install -e .
    ```

3. Adapt the data path in `config.yml` to point to the correct location on your
   machine.

#### With Docker

1. Build the Docker image locally:

    ```shell
    docker build -t tfl_training_sbi .
    ```

2. Start the container:

    ```shell
    docker run -it -p 8888:8888 tfl_training_sbi jupyter notebook
    ```

### Building Documentation

To create source code documentation, run:

```shell
bash build_scripts/build_docs.sh
```

Then, open `docs/build/html/index.html` in your browser. This will also rebuild
the Jupyter Book-based notebook documentation originally found in the `html`
directory.

Note: There is some non-trivial logic in the entrypoint that may collide with
mounting volumes directly inside `/home/jovyan/tfl_training_sbi`. If you want to
mount volumes there, the easiest way is to override the entrypoint or to mount
somewhere else and create a symbolic link. For details, see the `Dockerfile` and
`entrypoint.sh`.

---

Feel free to reach out if you have any questions or need assistance during the
training. Enjoy your learning experience!

## Collaborators

We would like to thank the The [Machine Learning ⇌ Science
Collaboratory](https://mlcolab.org/) and the [Mackelab, Machine Learning in
Science](https://www.mackelab.org/) for their collaboration on this training. They shared materials of previous SBI workshops with us and helped with the initial design of the new training. 
Both are part of the Cluster of Excellence - Machine Learning for Science at the
University of Tübingen.

![Figure](./notebooks/_static/images/logos_collaborators.png)

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]:
    https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
