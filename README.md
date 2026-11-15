# KNR4633 Assignment 1 — PCA Face Recognition

Give the program a cropped face photo and it returns an enrolled subject ID or `Unknown`. It uses principal component analysis (PCA) to turn each photo into a short list of numbers, then compares that list with those of enrolled faces. This is a classroom prototype evaluated on the Olivetti Faces dataset.

## Train the model

Open PowerShell in the `Project` folder. Use Python 3.11 or later. On a new machine, install the dependencies with `python -m pip install -r requirements.txt`. NumPy handles the arrays and PCA maths, Pillow reads and saves images, and Matplotlib makes the result figures.

```powershell
python face_recognition.py check-data
python face_recognition.py train
```

`check-data` confirms that the dataset has 400 photos and prints the split sizes. `train` fits the model on 180 photos from 30 enrolled subjects, uses validation photos to choose the component count and rejection threshold, and evaluates it on 110 held-out photos. It saves the model and results to `results`.

For another run, choose a new folder. The fixed seed gives each run the same data split.

```powershell
python face_recognition.py train --run-dir results_repeat
```

## Predict and check a photo

A **sample index** is the photo's position in the dataset, from 0 to 399. A **subject ID** is the person in the photo. Each subject has ten photos. For example, sample `346` is a photo of Subject `34`. Pick a `sample_index` from `results/predictions.csv` to try a held-out photo.

```powershell
python face_recognition.py predict --sample-index 346 --preview sample-346.png
Invoke-Item .\sample-346.png
```

The prediction prints the guessed subject or `Unknown`, the actual subject ID, and whether the guess was correct. If the person is outside the enrolment gallery, the expected answer is `Unknown`. The preview shows the grayscale input the model classified. Use a fresh filename each time, or omit `--preview` to print the result only.

Use `--model` to choose a saved model from another run. For your own photo, use `--image` with a cropped frontal face. You can check that prediction against the identity you know.

```powershell
python face_recognition.py predict --model .\results_repeat\model.npz --sample-index 346
python face_recognition.py predict --image ".\cropped-face.png"
```

## How the Python script uses the model

`face_recognition.py` contains the PCA and matching code. When you train, it calculates the average face and the principal directions, called **eigenfaces**. It saves these to `results/model.npz` along with the enrolled faces' coordinates, their subject IDs, and the rejection threshold.

When you predict, the Python script loads those saved values. It subtracts the average face from your photo and uses the eigenfaces to convert the 4,096 pixels into a short list of coordinates. It finds the nearest saved list in the gallery. The script returns that subject ID if the distance is within the threshold, or `Unknown` if the distance is too large. This uses the existing model from the selected run.

## Inspect the results

| File in the run folder | What to look at |
| --- | --- |
| `model.npz` | The saved NumPy arrays used for prediction. |
| `metrics.json` | The chosen component count, validation scores, and held-out results for PCA and the raw-pixel baseline. |
| `predictions.csv` | Each test photo's index, actual subject, predicted subject or `Unknown`, and nearest distance for both models. |
| `component_sweep.png` | Validation accuracy for 10, 25, 50, and 100 PCA components, alongside the cumulative explained variance plot. |
| `eigenfaces.png` | The average training face and the first seven eigenfaces. |
| `split_indices.npz` | The sample indices used for training, validation, and testing. |

To compare runs, open their `metrics.json` files. **Rank-1 accuracy** measures how often the closest gallery match is correct for enrolled subjects, before applying the threshold. The **unknown false acceptance rate** measures how often the system gives an enrolled subject ID to an unenrolled person.

## Project scripts

The project has one model script and three dataset scripts. Run the commands below from the `Project` folder. For the scripts in `scripts`, use `python -m` so Python can find the shared code in `face_recognition.py`.

### face_recognition.py

[face_recognition.py](face_recognition.py) is the main program. Its `check-data` command checks `olivetti_faces.npz`. Its `train` command fits PCA and saves `model.npz`, metrics, predictions, and figures in a run folder. Its `predict` command loads a saved model and classifies one photo. See [Train the model](#train-the-model) and [Predict and check a photo](#predict-and-check-a-photo) for the commands.

### export_dataset.py

Use [export_dataset.py](scripts/export_dataset.py) when you want to **view the full dataset as photos**. It reads `olivetti_faces.npz` and writes 400 grayscale PNGs plus `labels.csv` to a new folder. A name like `sample_346_subject_34.png` tells you both the photo's sample index and its subject ID.

```powershell
python -m scripts.export_dataset --output dataset_png_copy
```

The default output is `dataset_png`. To browse the existing export, open the [dataset_png folder](dataset_png) or its [labels.csv index](dataset_png/labels.csv).

### import_dataset.py

Use [import_dataset.py](scripts/import_dataset.py) when you want to **pack a folder of exported photos back into a NumPy archive**. It reads the PNGs and `labels.csv`. It checks that the CSV lists each sample index once and that each photo is a 64 × 64 grayscale PNG. The resulting `.npz` contains two arrays, `images` and `labels`, in the original sample order.

```powershell
python -m scripts.import_dataset --input dataset_png --output olivetti_faces_rebuilt.npz
```

The example uses the default input folder and output filename. The rebuilt pixel values reflect rounding to 8-bit PNGs. Keep the original `olivetti_faces.npz` for the assignment's training run.

### prepare_dataset.py

Use [prepare_dataset.py](scripts/prepare_dataset.py) when you need to **restore the original dataset from its source**. It uses scikit-learn to download Olivetti Faces, checks the image dimensions and subject counts, and writes `olivetti_faces.npz` in the Project folder. This regenerates the file if it already exists.

```powershell
python -m pip install scikit-learn
python -m scripts.prepare_dataset
```

If the download fails, save the source MAT file and pass its path with `python -m scripts.prepare_dataset --mat .\olivettifaces.mat`. The script checks the SHA-256 checksum before loading the file. [Source MAT file](https://ndownloader.figshare.com/files/5976027)

Image credit goes to AT&T Laboratories Cambridge. [Original Database of Faces](https://cam-orl.co.uk/facedatabase.html) · [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)

## Complete the report

Use the chosen run's metrics, individual predictions and figures to complete the Results, Discussion, and Conclusion in the Word draft. If you want Codex to finish the report, give it the name of the run folder you want to use. [Technical report draft](Assignment%201%20Technical%20Report.docx)
